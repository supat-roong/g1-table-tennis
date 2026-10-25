"""Self-play PPO with the duel and the policy on one GPU.

    python -m g1tt.gpu.train_gpu --tag g1 --max-hours 11
    python -m g1tt.gpu.train_gpu --tag g1 --max-hours 11 --resume

Same options, log rows and checkpoint format as g1tt.train, so runs move
between a laptop and a GPU machine: checkpoints are saved with CPU tensors and
resume anywhere. One iteration here is 2048 matches (98k decisions), 4.7 times a
CPU iteration, so the default 21000 iterations match the CPU's 100000.
--max-hours saves and stops before a session limit (Kaggle: 12 h); --resume
picks up from runs/<tag>/ckpt.pt.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from g1tt.env import ACTOR_DIM, CRITIC_DIM
from g1tt.gpu.duel_jax import TOT_KEYS, GpuCfg, GpuDuel
from g1tt.ppo import PPO, PPOCfg
from g1tt.train import RUNS, _rates, add_training_args, env_cfg_from


def to_torch(x):
  return torch.from_dlpack(x)


def to_jax(t):
  return jax.dlpack.from_dlpack(t.contiguous())


def _cpu(obj):
  if torch.is_tensor(obj):
    return obj.detach().cpu()
  if isinstance(obj, dict):
    return {k: _cpu(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return type(obj)(_cpu(v) for v in obj)
  return obj


class TorchPPO(PPO):
  """g1tt.ppo.PPO with its buffers and networks on a device, fed torch tensors."""

  def __init__(self, obs_dim, cobs_dim, act_dim, n_envs, cfg: PPOCfg, device):
    super().__init__(obs_dim, cobs_dim, act_dim, n_envs, cfg)
    self.device = device
    self.ac.to(device)
    self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.lr)
    self.buf = {k: v.to(device) for k, v in self.buf.items()}

  @torch.no_grad()
  def act(self, o, c):
    self.ac.obs_norm.update(o)
    self.ac.cobs_norm.update(c)
    d = self.ac.dist(o)
    a = d.sample()
    b = self.buf
    b["obs"][self.t], b["cobs"][self.t], b["act"][self.t] = o, c, a
    b["logp"][self.t] = d.log_prob(a).sum(-1)
    b["mu"][self.t], b["std"][self.t] = d.mean, d.stddev
    b["val"][self.t] = self.ac.value(c)
    return a

  @torch.no_grad()
  def record(self, rew, fell, cut, term_cobs):
    r = rew.float().clone()
    r = r + cut.float() * self.cfg.gamma * self.ac.value(term_cobs)  # cut short: credit where it was
    self.buf["rew"][self.t] = r
    self.buf["done"][self.t] = (fell | cut).float()
    self.t += 1
    return self.t == self.cfg.horizon

  def update(self, last_cobs):
    cfg, b = self.cfg, self.buf
    with torch.no_grad():
      last_v = self.ac.value(last_cobs)
      adv = torch.zeros_like(b["rew"])
      gae = torch.zeros(self.n, device=self.device)
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
      perm = torch.randperm(total, device=self.device)
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
          s0, s1 = std_old[idx], d.stddev
          kl = (torch.log(s1 / s0) + (s0 ** 2 + (mu_old[idx] - d.mean) ** 2) / (2 * s1 ** 2) - 0.5).sum(-1).mean()
          klv = kl.item()
          if klv > 2 * cfg.desired_kl:
            self.lr = max(1e-5, self.lr / 1.5)
          elif klv < cfg.desired_kl / 2:
            self.lr = min(1e-2, self.lr * 1.5)
          for g in self.opt.param_groups:
            g["lr"] = self.lr
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
        self.opt.step()
        stats["pi_loss"] += pi_loss.item()
        stats["v_loss"] += v_loss.item()
        stats["entropy"] += ent.item()
        stats["kl"] += klv
        count += 1
    self.t = 0
    out = {k: v / count for k, v in stats.items()}
    out["lr"] = self.lr
    out["std"] = self.std()
    return out

  def state_dict(self):
    return _cpu(super().state_dict())


def main():
  ap = argparse.ArgumentParser()
  add_training_args(ap)
  ap.set_defaults(iters=21000)
  ap.add_argument("--matches", type=int, default=2048, help="matches on the GPU (two robots each)")
  ap.add_argument("--impl", default="warp", help="MJX physics: warp (NVIDIA GPU) or jax")
  ap.add_argument("--plan-k", type=int, default=0, help="balls planned per step (0: matches / 4)")
  ap.add_argument("--runs", default=str(RUNS))
  ap.add_argument("--max-hours", type=float, default=None, help="save and stop after this long")
  args = ap.parse_args()

  t_start = time.time()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(args.seed)
  out = Path(args.runs) / args.tag
  out.mkdir(parents=True, exist_ok=True)

  env_cfg = env_cfg_from(args)
  env = GpuDuel(GpuCfg(n=args.matches, env=env_cfg, impl=args.impl, plan_k=args.plan_k, seed=args.seed))
  ppo_cfg = PPOCfg()
  n_agents = 2 * args.matches
  algo = TorchPPO(ACTOR_DIM, CRITIC_DIM, 29, n_agents, ppo_cfg, device)

  start, level = 0, args.level
  if args.resume and (out / "ckpt.pt").exists():
    sd = torch.load(out / "ckpt.pt", weights_only=False, map_location=device)
    algo.load_state_dict(sd["ppo"])
    start, level = sd["iter"] + 1, sd.get("level", level)
    print(f"resumed from iter {start} at serve level {level}", flush=True)
  elif args.init:
    sd = torch.load(args.init, weights_only=False, map_location="cpu")
    algo.ac.load_state_dict({k: v.to(device) for k, v in sd["ppo"]["ac"].items()})
    if args.init_std is not None:
      with torch.no_grad():
        algo.ac.log_std.fill_(float(np.log(args.init_std)))
    algo.lr = 2e-5
    for g in algo.opt.param_groups:
      g["lr"] = algo.lr
    print(f"initialised from {args.init}", flush=True)
  if level != env_cfg.level:
    env.set_level(level)
  (out / "cfg.json").write_text(json.dumps({"env": asdict(env_cfg), "ppo": asdict(ppo_cfg), "args": vars(args),
                                            "device": str(device), "jax": str(jax.devices())}, indent=1, default=str))

  key = jax.random.PRNGKey(args.seed + start)
  key, k0 = jax.random.split(key)
  S = env.init(k0)
  step = jax.jit(env.step, donate_argnums=(1,))
  S, obs_j, cobs_j = jax.jit(env.observe)(env.C, S)
  ep_ret = torch.zeros(n_agents, device=device)
  ep_len = torch.zeros(n_agents, device=device)
  done_ret, done_len = [], []
  term_acc, tot_acc = {}, {k: 0.0 for k in TOT_KEYS}
  steps, good_logs = 0, 0
  t0 = time.time()
  log = open(out / "log.jsonl", "a")
  try:
    for it in range(start, args.iters):
      t_it = time.time()
      while True:
        o, c = to_torch(obs_j), to_torch(cobs_j)
        a = algo.act(o, c)
        key, sub = jax.random.split(key)
        S, res = step(env.C, S, to_jax(a), sub)
        obs_j, cobs_j = res["obs"], res["cobs"]
        rew, fell, cut = to_torch(res["rew"]), to_torch(res["fell"]), to_torch(res["cut"])
        for k, v in res["terms"].items():
          term_acc[k] = term_acc.get(k, 0.0) + v
        for k in TOT_KEYS:
          tot_acc[k] = tot_acc[k] + res["tot"][k]
        ep_ret += rew
        ep_len += 1
        done = fell | cut
        if bool(done.any()):
          done_ret.append(ep_ret[done].mean().item())
          done_len.append(ep_len[done].mean().item())
          ep_ret[done] = 0
          ep_len[done] = 0
        steps += n_agents
        if algo.record(rew, fell, cut, to_torch(res["term_cobs"])):
          break
      t_roll = time.time() - t_it
      upd = algo.update(to_torch(cobs_j))

      if it % args.log_every == 0 or it == args.iters - 1:
        n_steps = args.log_every * ppo_cfg.horizon
        s = {k: float(v) for k, v in tot_acc.items()}
        if env.sgs is not None:
          for name in ("serve", "ret"):
            probs, p, seen = env._sgs_probs(S, name)
            vis = np.asarray(seen) > 0
            pp = np.asarray(p)
            s[f"sgs_{name}_visited"] = float(vis.mean())
            s[f"sgs_{name}_success"] = float(pp[vis].mean()) if vis.any() else 0.0
            s[f"sgs_{name}_frontier"] = float(((pp > 0.2) & (pp < 0.8) & vis).mean())
          s["sgs_workers"] = 1.0
        row = {"iter": it, "steps": steps, "time": round(time.time() - t0, 1),
               "fps": round(n_agents * ppo_cfg.horizon / (time.time() - t_it)), "rollout_s": round(t_roll, 2),
               "ep_return": float(np.mean(done_ret)) if done_ret else None,
               "ep_len_s": float(np.mean(done_len)) * 0.02 if done_len else None,
               **{k: float(f"{v:.4g}") for k, v in upd.items()}, "level": level,
               **{k: round(v, 4) for k, v in _rates(s).items()},
               "terms": {k: round(float(v) / n_steps, 4) for k, v in term_acc.items()}}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"it {it:5d} fps {row['fps']:6d} ret {row['ep_return'] or 0:8.2f} len {row['ep_len_s'] or 0:5.2f}s "
              f"lvl {level:.2f} hit {row['hit_rate']:.3f} land {row['land_rate']:.3f} imp_err {row['impact_pos_err']:.3f} "
              f"ref {row['ref_rmse']:.3f} rally {row['rally_mean']:.2f} bhp {row['bh_playable']:.2f} "
              f"std {row['std']:.3f} kl {row['kl']:.4f} lr {row['lr']:.1e}"
              + (f" sgs {row['sgs_ret_success']:.2f}/{row['sgs_ret_frontier']:.2f}" if "sgs_ret_success" in row else ""),
              flush=True)
        done_ret, done_len, term_acc = [], [], {}
        tot_acc = {k: 0.0 for k in TOT_KEYS}
        good_logs = good_logs + 1 if row["hit_rate"] >= args.promote_hit else 0
        if good_logs >= 3 and level < 1.0:
          level = min(1.0, level + args.level_step)
          env.set_level(level)
          S = env.reset_sgs_serve(S)
          good_logs = 0
          print(f"serve level -> {level}", flush=True)

      last = it == args.iters - 1
      out_of_time = args.max_hours is not None and time.time() - t_start > args.max_hours * 3600
      if (it + 1) % args.save_every == 0 or last or out_of_time:
        sd = {"ppo": algo.state_dict(), "iter": it, "env_cfg": asdict(env_cfg), "level": level}
        torch.save(sd, out / "ckpt.pt")
        if (it + 1) % args.keep_every == 0 or last:
          torch.save(sd, out / f"ckpt_{it + 1:05d}.pt")
      if out_of_time:
        print(f"stopping at iter {it}: --max-hours {args.max_hours} reached; resume with --resume", flush=True)
        break
  finally:
    log.close()


if __name__ == "__main__":
  main()
