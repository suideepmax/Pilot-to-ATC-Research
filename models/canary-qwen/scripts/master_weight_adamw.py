"""AdamW with fp32 master weights and fp32 optimizer state, for use under
Lightning's `precision: 16-true` (no GradScaler, no fp32 master weights,
verified via nemo/canary_ft's installed lightning/pytorch/plugins/precision/half.py).

Root cause this fixes (see research_log/ISSUES.md ISS-011): under plain
`torch.optim.AdamW` on fp16 params, `exp_avg_sq` underflows to exactly zero
for realistic gradient magnitudes (~2e-5), collapsing the update to
`-lr/eps * exp_avg` -- i.e. SGD-with-momentum at an LR of `lr/eps`, not
AdamW. This also makes `weight_decay` numerically inert in fp16, since
`1 - lr*wd` rounds to exactly 1.0 for any `lr*wd < ~4.9e-4`.

This optimizer keeps fp32 shadow copies of params, exp_avg, and exp_avg_sq,
so the AdamW math is done at full precision regardless of the model's
fp16 storage dtype. Works with FSDP2 DTensor parameters since all ops here
are elementwise (no rank-changing reshapes).

Loss-scale handling: gradients arriving in `p.grad` are assumed to be
computed from a loss multiplied by `loss_scale` upstream (see
salm_train_stable.py's StableSALM.training_step). `loss_scale` is a mutable
attribute so a caller (e.g. an on_before_optimizer_step hook) can
implement dynamic loss scaling (halve on overflow, grow after N clean
steps) without reconstructing the optimizer.

Gradient clipping (ISS-011/ISS-012 follow-up): this optimizer divides by
`grad_divisor`, NOT `loss_scale` directly, in its fp32 arithmetic below.
`grad_divisor` defaults to `loss_scale` (plain unscaling, no clipping) but a
caller can fold a global-norm clip coefficient in by setting
`grad_divisor = loss_scale / clip_coef` before calling step() -- this makes
gradient clipping happen as part of this optimizer's existing fp32 divide,
rather than as a separate operation on the raw fp16 gradient tensor (which
is what silently zeroed gradients under FSDP2/DTensor -- see ISS-012:
Lightning's native clip path squares each rank's local gradient-shard norm
IN FP16 before all-reducing, overflowing to inf at realistic magnitudes,
collapsing the clip coefficient to 0). Never apply a clip coefficient by
multiplying an already-fp16 gradient tensor -- verified empirically that
this risks catastrophic underflow instead (typical coefficients here are
~1e-3, which collapses most fp16 elements to exactly zero on write-back).
"""

import itertools
import warnings
from collections import defaultdict

import torch


class MasterWeightAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, loss_scale=1024.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.loss_scale = float(loss_scale)
        # Defaults to loss_scale (no clipping) until a caller sets it based on
        # a real global-norm measurement; see class docstring.
        self.grad_divisor = float(loss_scale)
        self._had_overflow_this_step = False

    # --- Persistent numerical state (survives resume) --------------------
    # loss_scale/grad_divisor are the dynamic loss-scale state, mutated every
    # step by StableSALM's on_before_optimizer_step (backoff/growth). These
    # are genuinely persistent state -- deliberately distinct from transient
    # per-step flags like _had_overflow_this_step (below), which are correct
    # to reset on every resume and are NOT included here.
    #
    # ISS-014 (2026-09-10, root-caused via a minimal non-distributed repro,
    # scratchpad/iss014_repro_realistic.py): torch.optim.Optimizer's default
    # load_state_dict() silently downcasts EVERY floating-point per-param
    # state tensor to the owning parameter's dtype
    # (Optimizer._process_value_according_to_param_policy:
    # `if param.is_floating_point(): return value.to(dtype=param.dtype, ...)`).
    # Since this optimizer's params are fp16, that means state["master"],
    # state["exp_avg"], state["exp_avg_sq"] -- saved as fp32 by design, the
    # entire point of this class -- get silently cast back to fp16 on every
    # single resume, regardless of this file's own state_dict()/step() logic.
    # Reproduced exactly: exp_avg_sq ~2e-12 (realistic per-element magnitude
    # for this model) underflows to fp16 0.0; eps=1e-8 also underflows to
    # fp16 0.0; denom = sqrt(0)+0 = 0; addcdiv_ then divides by exactly zero,
    # producing -inf in the very first resumed step -- this is what actually
    # corrupted step=41-last.ckpt in the smoke test (206 NaN + 265 Inf
    # tensors), not the loss_scale/grad_divisor omission (which was real but
    # numerically harmless in that specific run) and not anything
    # DTensor/two-group-specific. A post-hoc cast back to fp32 CANNOT undo
    # this -- the precision (and the exact-zero values) are already lost by
    # the time Python code could intervene -- so the fix must prevent the
    # downcast from happening at all, by not routing these three keys through
    # Optimizer.load_state_dict()'s per-param casting at all.
    _FP32_STATE_KEYS = ("master", "exp_avg", "exp_avg_sq")

    def state_dict(self):
        sd = super().state_dict()
        sd["loss_scale"] = self.loss_scale
        sd["grad_divisor"] = self.grad_divisor
        return sd

    def load_state_dict(self, state_dict):
        # Deliberately does NOT call super().load_state_dict() -- that method
        # (torch.optim.Optimizer.load_state_dict) is exactly what silently
        # downcasts master/exp_avg/exp_avg_sq to fp16 (see ISS-014 note
        # above), and there is no way to prevent that via a pre/post hook
        # since the cast is unconditional on param.is_floating_point(), not
        # gated by the incoming tensor's own dtype. This reimplements the
        # same structural validation and param-id remapping the base class
        # does, but preserves fp32 for this optimizer's own state tensors.
        state_dict = dict(state_dict)  # shallow copy; do not mutate caller's dict
        # Explicit compatibility policy for checkpoints saved before this fix
        # (no loss_scale/grad_divisor keys) -- warn loudly rather than
        # silently treating a partial resume as if it were exact.
        if "loss_scale" not in state_dict:
            warnings.warn(
                "MasterWeightAdamW.load_state_dict: checkpoint has no saved 'loss_scale' "
                "(pre-ISS-014 checkpoint) -- falling back to this run's configured default "
                f"({self.loss_scale}). This is NOT an exact resume of the dynamic loss-scale "
                "state.", stacklevel=2,
            )
        loss_scale = state_dict.pop("loss_scale", self.loss_scale)
        grad_divisor = state_dict.pop("grad_divisor", self.grad_divisor)

        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        for g, sg in zip(groups, saved_groups):
            if len(g["params"]) != len(sg["params"]):
                raise ValueError(
                    "loaded state dict contains a parameter group that doesn't match "
                    "the size of optimizer's group"
                )

        id_map = dict(zip(
            itertools.chain.from_iterable(g["params"] for g in saved_groups),
            itertools.chain.from_iterable(g["params"] for g in groups),
        ))

        new_state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k not in id_map:
                new_state[k] = v  # unassociated state, kept as-is (base class does the same)
                continue
            param = id_map[k]
            entry = {}
            for key, val in v.items():
                if key in self._FP32_STATE_KEYS and torch.is_tensor(val):
                    # Preserve fp32 exactly. Deliberately do NOT also pass
                    # device=param.device here (an earlier version of this
                    # fix did, and crashed under FSDP2/DTensor with
                    # "Expected all tensors to be on the same device, but
                    # found at least two devices, cuda:0 and cuda:N" on every
                    # non-rank-0 process -- found via a real 4-GPU smoke
                    # test, not assumed). By the time this method runs, the
                    # incoming tensor has already been placed onto this
                    # rank's own local DTensor shard by the upstream
                    # distributed-checkpoint load machinery; forcing a
                    # device move here can be actively wrong (it moved
                    # ranks 1-3's already-correct local state onto rank 0's
                    # device). Only change dtype if it isn't already fp32.
                    entry[key] = val if val.dtype == torch.float32 else val.to(dtype=torch.float32)
                else:
                    # step/other keys: keep exactly as provided by the
                    # upstream load, matching this same device-safety
                    # reasoning -- no dtype or device change.
                    entry[key] = val
            new_state[param] = entry

        new_param_groups = []
        for g, sg in zip(groups, saved_groups):
            ng = dict(sg)
            ng["params"] = g["params"]
            new_param_groups.append(ng)

        self.__setstate__({"state": new_state, "param_groups": new_param_groups})
        self.loss_scale = float(loss_scale)
        self.grad_divisor = float(grad_divisor)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._had_overflow_this_step = False

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # No per-parameter torch.isfinite(...).all() check here: on a
                # sharded DTensor grad that .all() is itself a collective, and
                # `continue`-ing past it only on affected ranks caused an NCCL
                # hang (ISS-011 follow-up). The caller (StableSALM's
                # on_before_optimizer_step) already does one global, correctly
                # all-reduced finiteness check and zeroes all grads on every
                # rank identically before this ever runs, so grads reaching
                # this optimizer are always already finite.
                grad = p.grad.detach()
                grad = grad.float() / self.grad_divisor

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["master"] = p.detach().clone().float()
                    state["exp_avg"] = torch.zeros_like(state["master"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master"])

                state["step"] += 1
                step = state["step"]
                master = state["master"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if wd != 0.0:
                    master.mul_(1 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                step_size = lr / bias_correction1

                master.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(master.to(p.dtype))

        return loss
