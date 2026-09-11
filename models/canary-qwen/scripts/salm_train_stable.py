"""Training entrypoint for full-decoder-scope Canary-Qwen fine-tuning under
`precision: 16-true` with real numerical stability (see research_log/ISSUES.md
ISS-011, ISS-012). A copy of NeMo's salm_train.py, except the model is a
StableSALM subclass that:

1. Scales the loss by the optimizer's current (dynamic) loss_scale before
   backward, so fp16 gradients don't underflow to zero before
   MasterWeightAdamW ever sees them.
2. In on_before_optimizer_step (which fires before optimizer.step()), checks
   each gradient tensor for non-finite values via a single fp32 all-reduce.
   If any rank sees one, every parameter's .grad is set to None (a true skip
   -- MasterWeightAdamW's `if p.grad is None: continue` then leaves
   step/momentum/master state completely untouched) and the loss scale is
   halved. After GROWTH_INTERVAL consecutive clean steps it is doubled back,
   mirroring torch.amp.GradScaler's own backoff/growth policy.
3. On clean steps, computes a gradient-clipping coefficient from that SAME
   fp32 all-reduced norm and folds it into MasterWeightAdamW's existing fp32
   divide (`grad_divisor`). Gradient clipping is NOT delegated to Lightning's
   native clip_gradients() -- ISS-012 found that path squares each rank's
   local gradient-shard norm IN FP16 before all-reducing (DTensor's
   _NormPartial), overflowing to inf at realistic gradient magnitudes and
   silently zeroing every gradient. `configure_gradient_clipping` is
   intentionally not overridden (the YAML leaves gradient_clip_val unset, so
   Lightning's default hook is already a no-op).
4. Logs grad norm / clip coefficient / skip / loss_scale via logging.info on
   rank 0 every step -- `trainer.logger: False` in the YAML means self.log()
   calls are never persisted anywhere queryable, so this is the only record
   of these values.
5. Verifies, once, that the first non-skipped optimizer step actually
   produced a nonzero exp_avg somewhere (on_before_zero_grad) -- a tripwire
   for the exact class of bug ISS-012 was: gradients measured as finite and
   nonzero, then silently zeroed before the optimizer ever used them.

MasterWeightAdamW (in master_weight_adamw.py, same directory) divides the
received gradient by its own live `grad_divisor` attribute (loss_scale, or
loss_scale/clip_coef on a clean step) and does the real AdamW math in fp32.
That attribute is the single source of truth for what the optimizer divides
by; this file never keeps its own separate copy.

Usage: identical to salm_train.py, but pass a config using
master_weight_adamw.MasterWeightAdamW as the optimizer _target_.
"""

import json
import os
import re
import time

import hydra
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from lightning.pytorch.callbacks import ModelCheckpoint

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.collections.speechlm2.parts.optim_setup import freeze_and_subset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# ISS-012 (2026-09-10): gradient clipping used to be delegated to Lightning's
# native self.clip_gradients() -> Precision.clip_grad_by_norm ->
# torch.nn.utils.clip_grad_norm_, which operates directly on raw fp16 p.grad
# tensors with NO fp32 upcast. Under FSDP2/DTensor, the per-rank local-shard
# norm is squared IN FP16 before the cross-rank all-reduce
# (torch/distributed/tensor/_ops/_math_ops.py's _NormPartial), which overflows
# to inf whenever a rank's local gradient-shard norm exceeds sqrt(65504)=~256
# -- routinely true at our observed gradient magnitudes (200-700) once
# multiplied by any loss_scale >= ~256. An inf norm collapses the clip
# coefficient to exactly 0, silently zeroing every gradient, AFTER
# on_before_optimizer_step's finiteness check already passed (that check only
# verifies the INPUT is finite; clipping then destroys it afterward,
# invisibly). Checkpoint inspection proved this actually happened: Gate 1 v8,
# Gate 2 v2, and Gate 2 v3 all performed EXACTLY ZERO weight updates (exp_avg/
# exp_avg_sq all zero, saved weights bit-identical to pretrained init).
#
# Fixed here: clipping is no longer a separate Lightning-native call at all.
# configure_gradient_clipping is removed (Lightning's default checks
# self.trainer.gradient_clip_val, which the YAML leaves unset/0, so the
# default hook is already a no-op -- no override needed). Instead,
# on_before_optimizer_step computes the clip coefficient from the SAME fp32
# all-reduced norm it already correctly computes (never touching a raw fp16
# tensor with it), and folds it into MasterWeightAdamW's existing fp32 divide
# via `grad_divisor = loss_scale / clip_coef` -- so the fp16 gradient tensor
# is only ever divided by one fp32 scalar, in fp32, inside the optimizer.
# This also avoids the underflow risk of the opposite naive fix (multiplying
# a small fp16 gradient by a small fp32 clip_coef and writing the result back
# to fp16 -- verified empirically: coefficients around 1e-3, typical at our
# gradient magnitudes vs a strict clip threshold, collapse most fp16 elements
# to exactly zero on that write-back).
GRADIENT_CLIP_VAL = 1000.0

# ISS-011 follow-up #3 / ISS-012 / ISS-013 (2026-09-10): perception.proj (a
# plain nn.Linear(1024, 2048), the only path from the frozen Canary encoder
# into Qwen3's embedding space) is randomly initialized in every config this
# project uses, including v1/v3 -- see ISS-013, this project never loads the
# released canary-qwen-2.5b checkpoint that DOES contain a trained version of
# this layer. v1/v3 successfully trained this same randomly-initialized layer
# via LoRA at lr=5e-4 and reached real, verified WER results, which is
# evidence a random bridge CAN be trained successfully at that LR -- but it is
# NOT evidence that 5e-4 specifically is necessary or optimal for this scope
# (full-decoder FT), since the flat-val_loss observation that originally
# motivated this specific value turned out (ISS-012) to be an artifact of
# zero gradients ever reaching ANY parameter, not a bridge-LR-mismatch signal.
# Kept as a plausible, low-risk optimization (matches the one data point where
# this composition demonstrably learned) but not to be presented as proven
# necessary until a real (non-zeroed) run's learning curve says otherwise.
BRIDGE_PARAM_PATTERN = r"^perception\.proj\..+$"
BRIDGE_LR = 5e-4

# Dynamic loss-scale backoff/growth, same style as torch.amp.GradScaler
# (backoff_factor=0.5, growth_factor=2.0) but with a much shorter
# growth_interval (200 vs GradScaler's default 2000) so the scale can
# actually calibrate within the short step counts used by this staged
# validation protocol's smoke tests -- reconsider raising it back toward
# 2000 once this trains at full (10000-step) scale.
LOSS_SCALE_BACKOFF_FACTOR = 0.5
LOSS_SCALE_GROWTH_FACTOR = 2.0
LOSS_SCALE_GROWTH_INTERVAL = 200
MIN_LOSS_SCALE = 1.0
MAX_LOSS_SCALE = 65536.0


class StableSALM(SALM):
    def configure_optimizers(self):
        # ISS-011 follow-up #3: split perception.proj (randomly-initialized
        # bridge layer, see BRIDGE_LR comment above) into its own param group
        # at BRIDGE_LR, everything else trainable stays at model.cfg.optimizer.lr.
        # Modeled on NeMo's own configure_optimizers_exclude_norm_from_wd
        # (nemo/collections/speechlm2/parts/optim_setup.py), which uses the
        # same freeze_and_subset + id(param)-based re-grouping pattern.
        trainable_ids = {id(p) for p in freeze_and_subset(
            self.named_parameters(),
            exclude_patterns=self.cfg.get("freeze_params", []),
            keep_patterns=self.cfg.get("prevent_freeze_params", []),
        )}
        bridge_pattern = re.compile(BRIDGE_PARAM_PATTERN)
        bridge_params, rest_params = [], []
        for name, p in self.named_parameters():
            if id(p) not in trainable_ids:
                continue
            (bridge_params if bridge_pattern.match(name) else rest_params).append(p)

        if not bridge_params:
            logging.warning(
                f"configure_optimizers: no trainable params matched BRIDGE_PARAM_PATTERN="
                f"{BRIDGE_PARAM_PATTERN!r} -- the bridge-layer LR split is inactive this run "
                f"(all trainable params will use model.cfg.optimizer.lr)."
            )
        # Hydra-overridable (model.optimizer.bridge_lr) so the bridge-LR-split
        # hypothesis (ISS-011 follow-up #3 / ISS-012) can be ablated via CLI
        # override (e.g. model.optimizer.bridge_lr=1e-5, matching the base LR,
        # to test whether the split matters at all) without a code change.
        # Falls back to the BRIDGE_LR module constant if not set in config.
        bridge_lr = self.cfg.optimizer.get("bridge_lr", BRIDGE_LR)
        logging.info(
            f"configure_optimizers: {len(bridge_params)} bridge params @ lr={bridge_lr}, "
            f"{len(rest_params)} other trainable params @ lr={self.cfg.optimizer.lr} (default)."
        )

        # hydra.utils.instantiate below passes self.cfg.optimizer's fields as
        # kwargs to MasterWeightAdamW.__init__, which has no `bridge_lr`
        # parameter -- must exclude it from the instantiated kwargs while
        # still being able to read it above via .get().
        optimizer_cfg = {k: v for k, v in self.cfg.optimizer.items() if k != "bridge_lr"}
        optim_groups = [
            {"params": bridge_params, "lr": bridge_lr},
            {"params": rest_params},
        ]
        optimizer = hydra.utils.instantiate(optimizer_cfg, optim_groups, _convert_="all")
        ans = {"optimizer": optimizer}
        if "lr_scheduler" in self.cfg:
            lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
            ans["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return ans

    def training_step(self, batch: dict, batch_idx: int):
        ans = super().training_step(batch, batch_idx)
        # ans["loss"] was already logged (unscaled, real units) inside
        # SALM.training_step via self.log_dict -- but that's also invisible,
        # since this config runs with trainer.logger: False (same gap
        # on_before_optimizer_step's grad-norm logging already worked around).
        # Print it directly so the gate-redesign's acceptance criterion
        # (train loss trending down, not just a noisy val_loss snapshot --
        # see ISS-011 follow-up #3) has something to check against.
        train_loss = ans["loss"].detach()
        if torch.distributed.get_rank() == 0 and batch_idx % 8 == 0:
            logging.info(f"training_step: batch_idx={batch_idx} train_loss_unscaled={train_loss.item():.6g}")
        # Only the RETURNED loss (used for backward) needs scaling. Read the
        # live scale off the optimizer (single source of truth) rather than
        # a local constant, since on_before_optimizer_step adjusts it every step.
        ans = dict(ans)
        ans["loss"] = ans["loss"] * self.optimizers().loss_scale
        return ans

    # ISS-012: configure_gradient_clipping is intentionally NOT overridden.
    # Lightning's default implementation calls self.clip_gradients() using
    # self.trainer.gradient_clip_val, which the YAML leaves unset (0/None) --
    # so the default hook is already a true no-op. Clipping now happens
    # entirely inside on_before_optimizer_step, folded into
    # MasterWeightAdamW's existing fp32 divide (see grad_divisor below),
    # never touching a raw fp16 gradient tensor with the clip coefficient.

    def on_before_optimizer_step(self, optimizer):
        # ISS-011 follow-up: an earlier version of this hook called
        # torch.isfinite(g).all() directly on DTensor gradients and `continue`d
        # past a later `.sum()` on non-finite ones. Both `.all()` and `.sum()`
        # are collective (ALLREDUCE) ops on a sharded DTensor -- skipping one
        # of them only on the ranks that saw a non-finite local shard silently
        # desynced which collective each rank expected next, hanging the job
        # (diagnosed via Opus consult, 2026-09-09). Fix: do all per-parameter
        # math on `.to_local()` (zero collectives, cannot desync), then issue
        # exactly ONE unconditional all-reduce per step, and branch only on
        # that already-global result -- never on a per-rank local value.
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]

        # ISS-012: this all-reduce is the ONLY place a global gradient norm is
        # ever computed, and it does so entirely in fp32 (gf = g.float()
        # below) -- unlike Lightning's native clip path, this cannot overflow
        # at realistic gradient magnitudes (fp32 max ~3.4e38).
        local_bad = torch.zeros((), device=self.device, dtype=torch.float32)
        local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for p in params:
            g = p.grad.detach()
            g = g.to_local() if isinstance(g, DTensor) else g
            gf = g.float()
            local_bad += (~torch.isfinite(gf)).any().to(torch.float32)
            local_sq += torch.nan_to_num(gf, nan=0.0, posinf=0.0, neginf=0.0).pow(2).sum()

        flags = torch.stack([local_bad, local_sq])
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)
        any_nonfinite = bool(flags[0].item() > 0.0)
        # loss_scale_used is captured BEFORE any mutation below, since it's the
        # scale that was actually in effect when this step's backward() ran
        # (ISS-012 logging fix: the previous version read optimizer.loss_scale
        # again after mutating it, so on backoff/growth steps the logged
        # scale didn't match the scale the norm was actually computed under).
        loss_scale_used = optimizer.loss_scale
        # flags[1] is the sum of squared *scaled* grads; on a skipped
        # (any_nonfinite) step this is contaminated by nan_to_num zeroing out
        # the nonfinite contributions, so it's NOT a real norm -- must not be
        # logged as one (ISS-012 logging fix #2: the previous version logged
        # this garbage value on skip steps, and DEC-009's GRADIENT_CLIP_VAL
        # calibration was partly derived from exactly these contaminated
        # entries).
        grad_norm_unscaled = float(flags[1].sqrt().item()) / loss_scale_used if not any_nonfinite else float("nan")

        # ISS-011 follow-up #2 (Opus consult, 2026-09-09): Gate 1 v6/v7 measured
        # a 14-20% non-finite rate under a *static* loss_scale, ~150-400x above
        # GradScaler's healthy <0.1% equilibrium, and it was LR-independent --
        # an artifact of a badly calibrated fixed scale, not real instability.
        # Two bugs in the earlier version of this method compounded that:
        #   1. p.grad.zero_() left p.grad non-None, so MasterWeightAdamW's
        #      `if p.grad is None: continue` didn't trigger -- state["step"]
        #      still incremented and exp_avg/exp_avg_sq still decayed, so a
        #      "skipped" step still applied a real (if attenuated) update
        #      from stale momentum instead of being a true no-op.
        #   2. loss_scale was a fixed module constant, never adjusted, so a
        #      persistently-too-high scale stayed too high forever.
        # Fixed here: skip by setting p.grad = None (true no-op, matches
        # torch.amp.GradScaler semantics of skipping optimizer.step() itself),
        # and adjust optimizer.loss_scale with the same backoff/growth policy
        # GradScaler uses, so the scale self-calibrates to whatever this
        # model's gradients actually tolerate.
        clip_coef = float("nan")
        if any_nonfinite:
            for p in params:
                p.grad = None
            optimizer.loss_scale = max(optimizer.loss_scale * LOSS_SCALE_BACKOFF_FACTOR, MIN_LOSS_SCALE)
            optimizer._consecutive_clean_steps = 0
            optimizer.grad_divisor = optimizer.loss_scale
        else:
            optimizer._consecutive_clean_steps = getattr(optimizer, "_consecutive_clean_steps", 0) + 1
            if optimizer._consecutive_clean_steps >= LOSS_SCALE_GROWTH_INTERVAL:
                optimizer.loss_scale = min(optimizer.loss_scale * LOSS_SCALE_GROWTH_FACTOR, MAX_LOSS_SCALE)
                optimizer._consecutive_clean_steps = 0

            # ISS-012: fold gradient clipping into the fp32 divide MasterWeightAdamW
            # already performs, instead of a separate fp16-native Lightning clip
            # call. clip_coef is computed from the fp32 grad_norm_unscaled above
            # (never from a raw fp16 tensor), so it cannot suffer the DTensor
            # local-norm-squared-in-fp16 overflow that caused ISS-012.
            # Equivalent to clipping the unscaled gradient to GRADIENT_CLIP_VAL,
            # then having the optimizer divide by loss_scale_used, in one fp32 op:
            # optimizer.step() will do grad.float() / grad_divisor, where
            # grad_divisor = loss_scale_used / clip_coef.
            #
            # gradient_clip_val_unscaled is now Hydra-overridable
            # (model.gradient_clip_val_unscaled) so a lowered threshold can be
            # tested (e.g. via percentile analysis of observed grad_norm_unscaled
            # in Gate 2/3 logs) without a code change and WITHOUT silently
            # changing behavior for any run that doesn't pass the override --
            # the module constant remains the default. A threshold change is a
            # candidate to be validated (does it actually reduce the observed
            # post-clip norm, does it change the loss trajectory), not an
            # established repair -- see research_log/DECISIONS.md.
            clip_val = self.cfg.get("gradient_clip_val_unscaled", GRADIENT_CLIP_VAL)
            clip_coef = min(1.0, clip_val / (grad_norm_unscaled + 1e-12))
            optimizer.grad_divisor = loss_scale_used / clip_coef

            # ISS-012 correctness tripwire: after the first real (non-skipped)
            # optimizer step, verify at least one trainable parameter's
            # exp_avg is actually nonzero -- i.e. a real update happened. This
            # exact class of bug (gradients silently zeroed between
            # measurement and use) went undetected for three full gate runs
            # because nothing checked this. Checked once, not every step.
            if not getattr(optimizer, "_verified_first_real_update", False):
                optimizer._pending_first_update_check = True

        # self.log() here is a no-op for calibration purposes: this config
        # runs with trainer.logger: False, so nothing ever persists these
        # values (confirmed absent from every Gate 1 run's artifacts). Print
        # via logging.info instead so they land in the training log file,
        # rank 0 only to avoid 4x duplicate lines.
        if torch.distributed.get_rank() == 0:
            logging.info(
                f"on_before_optimizer_step: grad_norm_unscaled={grad_norm_unscaled:.6g} "
                f"loss_scale_used={loss_scale_used:.1f} loss_scale_next={optimizer.loss_scale:.1f} "
                f"clip_coef={clip_coef:.6g} clip_val={self.cfg.get('gradient_clip_val_unscaled', GRADIENT_CLIP_VAL):.1f} "
                f"skipped={any_nonfinite}"
            )

    def on_before_zero_grad(self, optimizer):
        # ISS-012 correctness tripwire (part 2): fires after optimizer.step(),
        # so by now a real update (if any_nonfinite was False last step) has
        # already happened. Verify it actually did.
        if getattr(optimizer, "_pending_first_update_check", False):
            optimizer._pending_first_update_check = False
            # exp_avg is a DTensor (sharded, since it's built from p.detach().clone()
            # on an FSDP2-sharded parameter) -- torch.count_nonzero has no registered
            # DTensor sharding strategy, so this must operate on the LOCAL shard only
            # (.to_local()). This is a per-rank check, not a global all-reduce, and
            # that's fine for a tripwire: if THIS rank's local shard moved, gradients
            # reached the optimizer on this rank, which is exactly what's being
            # verified. Deliberately not adding another collective here.
            any_nonzero = any(
                torch.count_nonzero(
                    state["exp_avg"].to_local() if isinstance(state["exp_avg"], DTensor) else state["exp_avg"]
                ).item() > 0
                for state in optimizer.state.values()
                if "exp_avg" in state
            )
            if not any_nonzero:
                raise RuntimeError(
                    "ISS-012 tripwire: first non-skipped optimizer step completed but every "
                    "trainable parameter's exp_avg is still exactly zero -- gradients are being "
                    "lost somewhere between on_before_optimizer_step and MasterWeightAdamW.step(). "
                    "This is the exact failure class that silently invalidated Gate 1 v8 / Gate 2 "
                    "v2 / Gate 2 v3 (see research_log/ISSUES.md ISS-012). Do not proceed."
                )
            optimizer._verified_first_real_update = True
            if torch.distributed.get_rank() == 0:
                logging.info("ISS-012 tripwire: PASSED -- confirmed a real (nonzero) optimizer update occurred.")

    def on_validation_epoch_end(self):
        # GPT-6 Astra review (2026-09-10): Gate 3's early-stopping account was
        # initially wrong because Lightning's EarlyStopping only PRINTS on an
        # improving check ("Metric val_loss improved... New best score: X"),
        # not on a non-improving one -- so monitoring "improved" log lines
        # alone silently misses every non-improving check, and with
        # trainer.logger: False nothing persists them anywhere else either.
        # This writes EVERY validation check's result unconditionally, so the
        # full stopping trajectory can be reconstructed after the fact
        # instead of re-derived from an incomplete console-log audit.
        super().on_validation_epoch_end()
        if self.trainer.sanity_checking or torch.distributed.get_rank() != 0:
            return
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        # Bug found via direct resume smoke test (2026-09-10): if val_loss is
        # NaN, `val_loss < inf` evaluates to False (NaN compares False against
        # everything), so is_new_best stays False and _best_val_loss_seen is
        # never set on a fresh process -- then the unconditional access below
        # used to crash with AttributeError instead of recording the NaN.
        # Fixed: track best-so-far without ever assuming it was already set.
        prev_best = getattr(self, "_best_val_loss_seen", float("inf"))
        is_new_best = val_loss < prev_best
        self._best_val_loss_seen = val_loss if is_new_best else prev_best
        row = {
            "wall_time": time.time(),
            "global_step": int(self.trainer.global_step),
            "epoch": int(self.trainer.current_epoch),
            "val_loss": val_loss,
            "best_val_loss_so_far": float(self._best_val_loss_seen),
            "is_new_best": is_new_best,
            "lrs": [g["lr"] for g in self.optimizers().param_groups] if self.trainer.optimizers else None,
        }
        log_path = getattr(self, "_metrics_log_path", "val_metrics.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        logging.info(f"on_validation_epoch_end: {row}")


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    # Recovery checkpointing: exp_manager's own ModelCheckpoint (configured via
    # checkpoint_callback_params) is now purely a BEST-loss-monitored callback
    # (see salm_uwb_atcc_s3b3_fixed.yaml's every_n_train_steps/every_n_epochs
    # comment) -- it only saves on a genuine val_loss improvement, at whatever
    # step that occurs. That alone is not a substitute for periodic full-state
    # recovery: if training crashes between two improving checks (plausible
    # over a multi-hour run), there would be no recent checkpoint to resume
    # from at all. This second, UNMONITORED callback saves full trainer state
    # (weights + optimizer + scheduler + custom loss-scale/grad_divisor state,
    # since those live on the optimizer object which Lightning checkpoints
    # whole) every val_check_interval steps regardless of val_loss, keeping
    # only the single most recent one (save_top_k=1, monitor=None -> "most
    # recently saved" ordering) to bound disk usage.
    recovery_interval = int(cfg.trainer.get("val_check_interval", 250))
    recovery_ckpt = ModelCheckpoint(
        dirpath=str(log_dir / "recovery_checkpoints"),
        filename="{step}-recovery",
        monitor=None,
        every_n_train_steps=recovery_interval,
        save_top_k=1,
        save_last=False,
    )
    trainer.callbacks.append(recovery_ckpt)

    with trainer.init_module():
        model = StableSALM(OmegaConf.to_container(cfg.model, resolve=True))
    model._metrics_log_path = str(log_dir / "val_metrics.jsonl")

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
