"""Train the whole-body table tennis controller with PPO, by self-play on the CPU.

    .venv/bin/python -m g1tt.train --tag g1
    .venv/bin/python -m g1tt.train --tag g1 --resume

Two robots per match play each other with the same policy (g1tt.duel_env).
Writes runs/<tag>/: log.jsonl (one row per logged iteration), cfg.json,
ckpt.pt (latest, resumable) and ckpt_<iter>.pt snapshots. The GPU trainer
(g1tt.gpu.train_gpu) takes the same options and writes the same files.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from g1tt.env import ACTOR_DIM, CRITIC_DIM, EnvCfg
from g1tt.ppo import PPO, PPOCfg
from g1tt.vecenv import VecEnv

RUNS = Path(__file__).resolve().parents[2] / "runs"


def _rates(s):
  balls = max(s["hits"] + s["misses"], 1.0)
  return {
    "hit_rate": s["hits"] / balls,
    "land_rate": s["lands"] / balls,
    "falls": s["falls"],
    "episodes": s["episodes"],
    "balls_resolved": balls,
    "impact_pos_err": s["imp_pos_err"] / max(s["imp_n"], 1.0),
    "land_err": s["land_err"] / max(s["lands"], 1.0),
    "ref_rmse": s.get("ref_rmse", 0.0) / max(s.get("ref_n", 0.0), 1.0),
    "rally_mean": s.get("rally_sum", 0.0) / max(s.get("rallies", 0.0), 1.0),
    "shot_err": s.get("shot_err", 0.0) / max(s["hits"], 1.0),
    "shortfall": s.get("shortfall", 0.0) / max(s["hits"], 1.0),
    "bh_playable": s.get("bh_playable", 0.0) / max(s.get("bh_hits", 0.0), 1.0),  # backhand returns the partner can play
    # SGS: mean success over the return configurations and the share near the frontier (0.2-0.8)
    **{k: s[k] / s["sgs_workers"] for k in s if k.startswith("sgs_") and k != "sgs_workers"},
  }


def add_training_args(ap):
  """Options shared with the GPU trainer; the defaults are the recipe."""
  ap.add_argument("--tag", required=True)
  ap.add_argument("--iters", type=int, default=100000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--dr-scale", type=float, default=1.0)
  ap.add_argument("--resume", action="store_true")
  ap.add_argument("--init", default=None, help="start from another run's checkpoint (policy only)")
  ap.add_argument("--init-std", type=float, default=None, help="reset exploration noise after --init")
  ap.add_argument("--level", type=float, default=0.0, help="starting serve difficulty, 0..1")
  ap.add_argument("--level-step", type=float, default=0.25)
  ap.add_argument("--promote-hit", type=float, default=0.4, help="hit rate that raises the serve level")
  ap.add_argument("--move-free", type=float, default=0.10, help="metres off the spot allowed between shots")
  ap.add_argument("--wide-frac", type=float, default=0.10, help="share of returns aimed wide of the partner's reach")
  ap.add_argument("--bh-frac", type=float, default=0.25, help="share of returns aimed at the partner's backhand")
  ap.add_argument("--bh-shot-scale", type=float, default=2.0, help="weight of the shot and landing rewards on backhands")
  ap.add_argument("--ref-wrist", type=float, default=1.0, help="weight for following the human wrist on backhands")
  ap.add_argument("--playable", type=float, default=10.0, help="reward at the hit for a return the partner can play")
  ap.add_argument("--no-sgs", action="store_true", help="uniform serves and returns instead of Success Guided Sampling")
  ap.add_argument("--sgs-target", type=float, default=0.5, help="SGS: success rate it concentrates on")
  ap.add_argument("--sgs-kappa", type=float, default=5.0, help="SGS: concentration of the Beta kernel")
  ap.add_argument("--sgs-temp", type=float, default=2.0, help="SGS: softmax temperature (>= 1)")
  ap.add_argument("--sgs-history", type=int, default=100, help="SGS: outcomes kept per configuration")
  ap.add_argument("--sgs-mix", type=float, default=0.5, help="SGS: share of serves and returns it picks")
  ap.add_argument("--save-every", type=int, default=100, help="iterations between ckpt.pt saves")
  ap.add_argument("--keep-every", type=int, default=2000, help="iterations between kept ckpt_<iter>.pt snapshots")
  ap.add_argument("--log-every", type=int, default=5)


def env_cfg_from(args) -> EnvCfg:
  cfg = EnvCfg(dr_scale=args.dr_scale, level=args.level, move_free=args.move_free, wide_frac=args.wide_frac,
               bh_frac=args.bh_frac)
  cfg.sgs = None if args.no_sgs else {"target": args.sgs_target, "kappa": args.sgs_kappa,
                                      "temperature": args.sgs_temp, "history": args.sgs_history, "mix": args.sgs_mix}
  cfg.reward.bh_shot_scale = args.bh_shot_scale
  cfg.reward.playable = args.playable
  cfg.reward.ref_wrist = args.ref_wrist
  return cfg


def main():
  ap = argparse.ArgumentParser()
  add_training_args(ap)
  ap.add_argument("--workers", type=int, default=9)
  ap.add_argument("--envs", type=int, default=96, help="robots per worker")
  ap.add_argument("--threads", type=int, default=2, help="torch threads while workers step")
  ap.add_argument("--update-threads", type=int, default=8)
  args = ap.parse_args()

  torch.set_num_threads(args.threads)
  torch.manual_seed(args.seed)
  out = RUNS / args.tag
  out.mkdir(parents=True, exist_ok=True)

  env_cfg = env_cfg_from(args)
  ppo_cfg = PPOCfg()
  venv = VecEnv(args.workers, args.envs, seed=args.seed, cfg=env_cfg)
  algo = PPO(ACTOR_DIM, CRITIC_DIM, 29, venv.n, ppo_cfg)
  start = 0
  level = args.level
  if args.resume and (out / "ckpt.pt").exists():
    sd = torch.load(out / "ckpt.pt", weights_only=False)
    algo.load_state_dict(sd["ppo"])
    start = sd["iter"] + 1
    level = sd.get("level", level)
    print(f"resumed from iter {start} at serve level {level}")
  elif args.init:
    sd = torch.load(args.init, weights_only=False)
    algo.ac.load_state_dict(sd["ppo"]["ac"])
    if args.init_std is not None:
      with torch.no_grad():
        algo.ac.log_std.fill_(float(np.log(args.init_std)))
    # A fresh Adam takes full-size first steps, which wrecks a trained policy
    # at the default rate; start small and let the KL rule raise it.
    algo.lr = 2e-5
    for g in algo.opt.param_groups:
      g["lr"] = algo.lr
    print(f"initialised from {args.init}")
  if level != args.level:
    venv.set_level(level)
  good_logs = 0
  (out / "cfg.json").write_text(json.dumps({"env": asdict(env_cfg), "ppo": asdict(ppo_cfg), "args": vars(args)},
                                           indent=1, default=str))

  obs, cobs = venv.observe()
  ep_ret = np.zeros(venv.n)
  ep_len = np.zeros(venv.n)
  done_ret, done_len = [], []
  term_acc = {}
  t0 = time.time()
  steps = 0
  log = open(out / "log.jsonl", "a")
  try:
    for it in range(start, args.iters):
      t_it = time.time()
      while True:
        act = algo.act(obs, cobs)
        obs, cobs, rew, fell, timeout, term_cobs, terms = venv.step(act)
        for k, v in terms.items():
          term_acc[k] = term_acc.get(k, 0.0) + v
        ep_ret += rew
        ep_len += 1
        done = fell | timeout
        if done.any():
          done_ret.extend(ep_ret[done].tolist())
          done_len.extend(ep_len[done].tolist())
          ep_ret[done] = 0
          ep_len[done] = 0
        steps += venv.n
        if algo.record(rew, fell, timeout, term_cobs):
          break
      t_roll = time.time() - t_it
      # Workers idle during the update, so it can have the cores.
      torch.set_num_threads(args.update_threads)
      upd = algo.update(cobs)
      torch.set_num_threads(args.threads)

      if it % args.log_every == 0 or it == args.iters - 1:
        n_steps = args.log_every * ppo_cfg.horizon
        row = {
          "iter": it, "steps": steps, "time": round(time.time() - t0, 1),
          "fps": round(venv.n * ppo_cfg.horizon / (time.time() - t_it)),
          "rollout_s": round(t_roll, 2),
          "ep_return": float(np.mean(done_ret)) if done_ret else None,
          "ep_len_s": float(np.mean(done_len)) * 0.02 if done_len else None,
          **{k: float(f"{v:.4g}") for k, v in upd.items()},
          "level": level,
          **{k: round(v, 4) for k, v in _rates(venv.stats()).items()},
          "terms": {k: round(v / n_steps, 4) for k, v in term_acc.items()},
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"it {it:5d} fps {row['fps']:5d} ret {row['ep_return'] or 0:8.2f} len {row['ep_len_s'] or 0:5.2f}s "
              f"lvl {level:.2f} hit {row['hit_rate']:.3f} land {row['land_rate']:.3f} imp_err {row['impact_pos_err']:.3f} "
              f"ref {row['ref_rmse']:.3f} rally {row['rally_mean']:.2f} bhp {row['bh_playable']:.2f} "
              f"std {row['std']:.3f} kl {row['kl']:.4f} lr {row['lr']:.1e}"
              + (f" sgs {row['sgs_ret_success']:.2f}/{row['sgs_ret_frontier']:.2f}" if "sgs_ret_success" in row else ""),
              flush=True)
        done_ret, done_len, term_acc = [], [], {}
        # Serve curriculum: harder serves once these are being hit.
        good_logs = good_logs + 1 if row["hit_rate"] >= args.promote_hit else 0
        if good_logs >= 3 and level < 1.0:
          level = min(1.0, level + args.level_step)
          venv.set_level(level)
          good_logs = 0
          print(f"serve level -> {level}", flush=True)

      if (it + 1) % args.save_every == 0 or it == args.iters - 1:
        sd = {"ppo": algo.state_dict(), "iter": it, "env_cfg": asdict(env_cfg), "level": level}
        torch.save(sd, out / "ckpt.pt")
        if (it + 1) % args.keep_every == 0 or it == args.iters - 1:
          torch.save(sd, out / f"ckpt_{it + 1:05d}.pt")
  finally:
    log.close()
    venv.close()


if __name__ == "__main__":
  main()
