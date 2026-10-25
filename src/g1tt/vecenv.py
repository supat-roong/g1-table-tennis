"""Run several `DuelEnv` batches in worker processes.

Each worker owns `envs_per_worker` robots (half as many matches). Observations, rewards and actions
move through shared memory; a pipe only carries the "step" signal and the
small per-step info, so a step costs one round trip per worker.
"""

import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from g1tt.duel_env import DuelEnv
from g1tt.env import ACTOR_DIM, CRITIC_DIM, EnvCfg

FIELDS = {
  "obs": (ACTOR_DIM, np.float32),
  "cobs": (CRITIC_DIM, np.float32),
  "term_cobs": (CRITIC_DIM, np.float32),
  "act": (29, np.float32),
  "rew": (1, np.float32),
  "fell": (1, np.bool_),
  "timeout": (1, np.bool_),
}


def _views(shms, n):
  out = {}
  for k, (dim, dt) in FIELDS.items():
    out[k] = np.ndarray((n, dim), dtype=dt, buffer=shms[k].buf)
  return out


def _worker(conn, names, total, lo, hi, seed, cfg):
  shms = {k: shared_memory.SharedMemory(name=v) for k, v in names.items()}
  buf = _views(shms, total)
  env = DuelEnv(hi - lo, seed=seed, cfg=cfg)
  buf["obs"][lo:hi] = env.observe()
  buf["cobs"][lo:hi] = env.critic_obs()
  conn.send("ready")
  while True:
    cmd = conn.recv()
    if cmd == "step":
      obs, cobs, rew, fell, info = env.step(buf["act"][lo:hi])
      buf["obs"][lo:hi] = obs
      buf["cobs"][lo:hi] = cobs
      buf["term_cobs"][lo:hi] = info["term_cobs"]
      buf["rew"][lo:hi, 0] = rew
      buf["fell"][lo:hi, 0] = fell
      buf["timeout"][lo:hi, 0] = info["timeout"]
      conn.send(info["terms"])
    elif isinstance(cmd, tuple) and cmd[0] == "level":
      env.set_level(cmd[1])
      conn.send("ok")
    elif cmd == "stats":
      conn.send(env.pop_stats())
    elif cmd == "close":
      for s in shms.values():
        s.close()
      conn.send("bye")
      return


class VecEnv:
  def __init__(self, workers: int, envs_per_worker: int, seed: int = 0, cfg: EnvCfg | None = None):
    self.n = workers * envs_per_worker
    self.shms = {}
    for k, (dim, dt) in FIELDS.items():
      size = max(1, self.n * dim * np.dtype(dt).itemsize)
      self.shms[k] = shared_memory.SharedMemory(create=True, size=size)
    self.buf = _views(self.shms, self.n)
    names = {k: s.name for k, s in self.shms.items()}
    ctx = mp.get_context("spawn")
    self.conns, self.procs = [], []
    for w in range(workers):
      a, b = ctx.Pipe()
      lo, hi = w * envs_per_worker, (w + 1) * envs_per_worker
      p = ctx.Process(target=_worker, args=(b, names, self.n, lo, hi, seed * 1000 + w, cfg), daemon=True)
      p.start()
      self.conns.append(a)
      self.procs.append(p)
    for c in self.conns:
      assert c.recv() == "ready"

  def observe(self):
    return self.buf["obs"].copy(), self.buf["cobs"].copy()

  def step(self, actions: np.ndarray):
    self.buf["act"][:] = actions
    for c in self.conns:
      c.send("step")
    terms = [c.recv() for c in self.conns]
    b = self.buf
    mean_terms = {k: float(np.mean([t[k] for t in terms])) for k in terms[0]}
    return (b["obs"].copy(), b["cobs"].copy(), b["rew"][:, 0].copy(), b["fell"][:, 0].copy(),
            b["timeout"][:, 0].copy(), b["term_cobs"].copy(), mean_terms)

  def set_level(self, level: float):
    for c in self.conns:
      c.send(("level", level))
    for c in self.conns:
      c.recv()

  def stats(self):
    for c in self.conns:
      c.send("stats")
    out = {}
    for c in self.conns:
      for k, v in c.recv().items():
        out[k] = out.get(k, 0.0) + float(v)
    return out

  def close(self):
    for c in self.conns:
      try:
        c.send("close")
        c.recv()
      except (BrokenPipeError, EOFError):
        pass
    for p in self.procs:
      p.join(timeout=5)
    for s in self.shms.values():
      s.close()
      s.unlink()
