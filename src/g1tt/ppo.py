"""PPO with an asymmetric actor-critic.

The actor sees what a real robot would (proprioception and the planner's
command); the critic also sees privileged simulator state. Observations are
normalised with running statistics, advantages use GAE with value
bootstrapping on time-outs, and the learning rate adapts to a target KL.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class PPOCfg:
  horizon: int = 24
  epochs: int = 5
  minibatches: int = 4
  gamma: float = 0.99
  lam: float = 0.95
  clip: float = 0.2
  lr: float = 1e-3
  desired_kl: float = 0.01
  entropy: float = 0.0
  value_coef: float = 1.0
  max_grad_norm: float = 1.0
  init_std: float = 0.6
  max_std: float = 0.7  # exploration noise may shrink but not grow past this
  min_std: float = 0.05
  hidden: tuple = (512, 256, 128)


class RunningNorm(nn.Module):
  def __init__(self, dim: int, clip: float = 5.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))
    self.clip = clip

  @torch.no_grad()
  def update(self, x: torch.Tensor):
    b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = b_mean - self.mean
    tot = self.count + b_n
    self.mean += delta * b_n / tot
    self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
    self.count = tot

  def forward(self, x):
    return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(inp, hidden, out):
  layers, d = [], inp
  for h in hidden:
    layers += [nn.Linear(d, h), nn.ELU()]
    d = h
  layers.append(nn.Linear(d, out))
  return nn.Sequential(*layers)


class ActorCritic(nn.Module):
  def __init__(self, obs_dim, cobs_dim, act_dim, cfg: PPOCfg):
    super().__init__()
    self.obs_norm = RunningNorm(obs_dim)
    self.cobs_norm = RunningNorm(cobs_dim)
    self.actor = mlp(obs_dim, cfg.hidden, act_dim)
    self.critic = mlp(cobs_dim, cfg.hidden, 1)
    self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(cfg.init_std))))
    self.log_std_lo, self.log_std_hi = float(np.log(cfg.min_std)), float(np.log(cfg.max_std))
    with torch.no_grad():
      self.actor[-1].weight.mul_(0.01)
      self.actor[-1].bias.zero_()

  def dist(self, obs):
    mu = self.actor(self.obs_norm(obs))
    std = self.log_std.clamp(self.log_std_lo, self.log_std_hi).exp()
    return torch.distributions.Normal(mu, std.expand_as(mu))

  def value(self, cobs):
    return self.critic(self.cobs_norm(cobs)).squeeze(-1)

  @torch.no_grad()
  def act_deterministic(self, obs):
    return self.actor(self.obs_norm(obs))


class PPO:
  def __init__(self, obs_dim, cobs_dim, act_dim, n_envs, cfg: PPOCfg | None = None):
    self.cfg = cfg or PPOCfg()
    self.n = n_envs
    self.ac = ActorCritic(obs_dim, cobs_dim, act_dim, self.cfg)
    self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)
    self.lr = self.cfg.lr
    T = self.cfg.horizon
    z = lambda *s: torch.zeros(*s)
    self.buf = dict(obs=z(T, n_envs, obs_dim), cobs=z(T, n_envs, cobs_dim), act=z(T, n_envs, act_dim),
                    logp=z(T, n_envs), val=z(T, n_envs), rew=z(T, n_envs), done=z(T, n_envs),
                    mu=z(T, n_envs, act_dim), std=z(T, n_envs, act_dim))
    self.t = 0

  @torch.no_grad()
  def act(self, obs: np.ndarray, cobs: np.ndarray):
    o, c = torch.from_numpy(obs), torch.from_numpy(cobs)
    self.ac.obs_norm.update(o)
    self.ac.cobs_norm.update(c)
    d = self.ac.dist(o)
    a = d.sample()
    b = self.buf
    b["obs"][self.t], b["cobs"][self.t], b["act"][self.t] = o, c, a
    b["logp"][self.t] = d.log_prob(a).sum(-1)
    b["mu"][self.t], b["std"][self.t] = d.mean, d.stddev
    b["val"][self.t] = self.ac.value(c)
    return a.numpy()

  @torch.no_grad()
  def record(self, rew, fell, timeout, term_cobs):
    r = torch.from_numpy(rew).float()
    to = torch.from_numpy(timeout)
    if to.any():
      # The episode was cut, not ended: credit the value of where it was.
      r[to] += self.cfg.gamma * self.ac.value(torch.from_numpy(term_cobs[to.numpy()]))
    self.buf["rew"][self.t] = r
    self.buf["done"][self.t] = torch.from_numpy(fell | timeout).float()
    self.t += 1
    return self.t == self.cfg.horizon

  def update(self, last_cobs: np.ndarray):
    cfg, b = self.cfg, self.buf
    with torch.no_grad():
      last_v = self.ac.value(torch.from_numpy(last_cobs))
      adv = torch.zeros_like(b["rew"])
      gae = torch.zeros(self.n)
      for t in reversed(range(cfg.horizon)):
        nv = last_v if t == cfg.horizon - 1 else b["val"][t + 1]
        nonterm = 1.0 - b["done"][t]
        delta = b["rew"][t] + cfg.gamma * nv * nonterm - b["val"][t]
        gae = delta + cfg.gamma * cfg.lam * nonterm * gae
        adv[t] = gae
      ret = adv + b["val"]

    flat = lambda x: x.reshape(-1, *x.shape[2:])
    obs, cobs, act = flat(b["obs"]), flat(b["cobs"]), flat(b["act"])
    mu_old, std_old = flat(b["mu"]), flat(b["std"])
    logp_old, val_old, adv, ret = flat(b["logp"]), flat(b["val"]), flat(adv), flat(ret)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    total = obs.shape[0]
    mb = total // cfg.minibatches
    stats = dict(pi_loss=0.0, v_loss=0.0, entropy=0.0, kl=0.0)
    count = 0
    for _ in range(cfg.epochs):
      perm = torch.randperm(total)
      for i in range(cfg.minibatches):
        idx = perm[i * mb:(i + 1) * mb]
        d = self.ac.dist(obs[idx])
        logp = d.log_prob(act[idx]).sum(-1)
        ratio = torch.exp(logp - logp_old[idx])
        pi_loss = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
        v = self.ac.value(cobs[idx])
        v_clip = val_old[idx] + torch.clamp(v - val_old[idx], -cfg.clip, cfg.clip)
        v_loss = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
        ent = d.entropy().sum(-1).mean()
        loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy * ent

        with torch.no_grad():
          # Exact KL between the rollout policy and the current one.
          s0, s1 = std_old[idx], d.stddev
          kl = (torch.log(s1 / s0) + (s0**2 + (mu_old[idx] - d.mean) ** 2) / (2 * s1**2) - 0.5).sum(-1).mean()
          if kl > 2 * cfg.desired_kl:
            self.lr = max(1e-5, self.lr / 1.5)
          elif kl < cfg.desired_kl / 2:
            self.lr = min(1e-2, self.lr * 1.5)
          for g in self.opt.param_groups:
            g["lr"] = self.lr

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
        self.opt.step()
        stats["pi_loss"] += pi_loss.item()
        stats["v_loss"] += v_loss.item()
        stats["entropy"] += ent.item()
        stats["kl"] += kl.item()
        count += 1
    self.t = 0
    out = {k: v / count for k, v in stats.items()}
    out["lr"] = self.lr
    out["std"] = self.std()
    return out

  def std(self):
    return self.ac.log_std.clamp(self.ac.log_std_lo, self.ac.log_std_hi).exp().mean().item()

  def state_dict(self):
    return {"ac": self.ac.state_dict(), "opt": self.opt.state_dict(), "lr": self.lr, "cfg": asdict(self.cfg)}

  def load_state_dict(self, sd, load_opt: bool = True):
    self.ac.load_state_dict(sd["ac"])
    if load_opt:
      self.opt.load_state_dict(sd["opt"])
      self.lr = sd["lr"]

