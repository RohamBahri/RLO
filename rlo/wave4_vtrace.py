"""Undiscounted V-trace on decision trajectories, including forced-move reward.

Clipped corrections introduce bias; this is *not* an unbiased general off-policy
policy-gradient estimator. Bootstraps only when a sequence is truncated alive.
"""
import torch
import torch.nn.functional as F


def targets(rewards, values, behavior_logp, target_logp, *, bootstrap=0.,
            rho_bar=1., c_bar=1., pg_rho_bar=1.):
    rewards, values = rewards.float(), values.float()
    behavior_logp, target_logp = behavior_logp.float(), target_logp.float()
    if rewards.ndim != 1 or any(x.shape != rewards.shape for x in
                                (values, behavior_logp, target_logp)):
        raise ValueError("V-trace inputs must be equal-length 1D trajectories")
    if not len(rewards) or min(rho_bar, c_bar, pg_rho_bar) <= 0:
        raise ValueError("empty trajectory or invalid importance caps")
    with torch.no_grad():
        rho = (target_logp.detach() - behavior_logp.detach()).clamp(-50, 50).exp()
        clipped_rho = rho.clamp(max=rho_bar)
        cs = rho.clamp(max=c_bar)
        next_values = torch.cat([values.detach()[1:],
                                 values.detach().new_tensor([bootstrap])])
        delta = clipped_rho * (rewards + next_values - values.detach())
        vs = torch.empty_like(values)
        correction = values.new_zeros(())
        for t in range(len(rewards)-1, -1, -1):
            correction = delta[t] + cs[t] * correction
            vs[t] = values.detach()[t] + correction
        v_next = torch.cat([vs[1:], vs.new_tensor([bootstrap])])
        advantages = rho.clamp(max=pg_rho_bar) * (
            rewards + v_next - values.detach())
    return vs, advantages


def loss(rewards, values, behavior_logp, target_logp, *, bootstrap=0.,
         entropy=None, entropy_weight=0.):
    vt, advantages = targets(rewards, values, behavior_logp, target_logp,
                             bootstrap=bootstrap)
    policy = -(target_logp * advantages.detach()).mean()
    critic = .5 * F.mse_loss(values, vt.detach())
    regularizer = (-entropy_weight * entropy.mean() if entropy is not None
                   else values.new_zeros(()))
    return policy + critic + regularizer, dict(actor=float(policy.detach()),
                                               critic=float(critic.detach()))
