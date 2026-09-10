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
