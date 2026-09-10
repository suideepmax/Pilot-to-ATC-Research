"""Training entrypoint for full-decoder-scope Canary-Qwen fine-tuning under
`precision: 16-true` with real numerical stability (see research_log/ISSUES.md
ISS-011). A copy of NeMo's salm_train.py, except the model is a StableSALM
subclass that:

1. Scales the loss by the optimizer's current (dynamic) loss_scale before
   backward, so fp16 gradients don't underflow to zero before
   MasterWeightAdamW ever sees them.
2. Scales the configured gradient_clip_val by the same live factor before
   Lightning's clip_gradients runs, so the YAML's gradient_clip_val keeps
   its normal (unscaled) meaning.
3. In on_before_optimizer_step (which fires BEFORE clipping), checks each
   gradient tensor for non-finite values. If any rank sees one, every
   parameter's .grad is set to None (a true skip -- MasterWeightAdamW's
   `if p.grad is None: continue` then leaves step/momentum/master state
   completely untouched, unlike the earlier zero_()-based version which
   still advanced momentum on a "skipped" step) and the loss scale is
   halved. After GROWTH_INTERVAL consecutive clean steps it is doubled
   back, mirroring torch.amp.GradScaler's own backoff/growth policy.

   Gate 1 (v6/v7, 2026-09-09) measured a 14-20% non-finite rate under the
   previous *static* LOSS_SCALE=1024 -- 150-400x above GradScaler's healthy
   equilibrium (<0.1%, growth_interval=2000 by default) -- and the rate was
   LR-independent, i.e. an fp16-overflow property of the fixed scale itself,
   not a training-instability signal. Dynamic scaling lets the scale settle
   to whatever value this model's gradients actually tolerate instead of
   guessing one static value up front (see ISS-011 follow-up notes).
4. Logs grad norm / skip / current loss_scale via logging.info on rank 0
   every step -- `trainer.logger: False` in the YAML means self.log() calls
   are never persisted anywhere queryable, so this is the only record of
   these values (Gate 1 v6/v7 both ran with none captured).

MasterWeightAdamW (in master_weight_adamw.py, same directory) divides the
received gradient by its own live `loss_scale` attribute and does the real
AdamW math in fp32. That attribute is the single source of truth: this file
never keeps its own separate copy of the scale, always reading/writing
`optimizer.loss_scale` so training_step, configure_gradient_clipping, and
on_before_optimizer_step can never drift out of sync with each other.

Usage: identical to salm_train.py, but pass a config using
master_weight_adamw.MasterWeightAdamW as the optimizer _target_.
"""

import os

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor

from nemo.collections.speechlm2 import SALM, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Lightning's LightningModule.clip_gradients() raises MisconfigurationException
# if Trainer(gradient_clip_val=...) is ALSO set while we pass our own explicit
# value here -- it assumes double-configuration. So `gradient_clip_val` must
# be left unset (None) in the YAML's trainer block, and this constant is the
# real, single source of truth for the (unscaled) clip threshold.
GRADIENT_CLIP_VAL = 0.5

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
    def training_step(self, batch: dict, batch_idx: int):
        ans = super().training_step(batch, batch_idx)
        # ans["loss"] was already logged (unscaled, real units) inside
        # SALM.training_step via self.log_dict -- only the RETURNED loss
        # (used for backward) needs scaling. Read the live scale off the
        # optimizer (single source of truth) rather than a local constant,
        # since on_before_optimizer_step adjusts it every step.
        ans = dict(ans)
        ans["loss"] = ans["loss"] * self.optimizers().loss_scale
        return ans

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Ignore the passed-in / self.trainer.gradient_clip_val entirely --
        # Trainer(gradient_clip_val=...) must be left unset (None) in the YAML
        # for this override to be allowed to run at all (see GRADIENT_CLIP_VAL
        # comment above). GRADIENT_CLIP_VAL is the single source of truth for
        # the unscaled threshold; `optimizer` here is the raw MasterWeightAdamW
        # instance (Lightning passes it directly to this hook), so its
        # .loss_scale is always current.
        self.clip_gradients(
            optimizer,
            gradient_clip_val=GRADIENT_CLIP_VAL * optimizer.loss_scale,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm",
        )

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
        # flags[1] is the sum of squared *scaled* grads (scaled by the loss_scale
        # that was in effect when this step's backward() ran, i.e. the current
        # value below, since it hasn't been mutated yet this step).
        grad_norm_unscaled = float(flags[1].sqrt().item()) / optimizer.loss_scale

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
        if any_nonfinite:
            for p in params:
                p.grad = None
            optimizer.loss_scale = max(optimizer.loss_scale * LOSS_SCALE_BACKOFF_FACTOR, MIN_LOSS_SCALE)
            optimizer._consecutive_clean_steps = 0
        else:
            optimizer._consecutive_clean_steps = getattr(optimizer, "_consecutive_clean_steps", 0) + 1
            if optimizer._consecutive_clean_steps >= LOSS_SCALE_GROWTH_INTERVAL:
                optimizer.loss_scale = min(optimizer.loss_scale * LOSS_SCALE_GROWTH_FACTOR, MAX_LOSS_SCALE)
                optimizer._consecutive_clean_steps = 0

        # self.log() here is a no-op for calibration purposes: this config
        # runs with trainer.logger: False, so nothing ever persists these
        # values (confirmed absent from every Gate 1 run's artifacts). Print
        # via logging.info instead so they land in the training log file,
        # rank 0 only to avoid 4x duplicate lines.
        if torch.distributed.get_rank() == 0:
            logging.info(
                f"on_before_optimizer_step: grad_norm_unscaled={grad_norm_unscaled:.6g} "
                f"loss_scale={optimizer.loss_scale:.1f} skipped={any_nonfinite}"
            )


@hydra_runner(config_path="conf", config_name="salm")
def train(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = StableSALM(OmegaConf.to_container(cfg.model, resolve=True))

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
