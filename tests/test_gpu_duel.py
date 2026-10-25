"""The JAX duel computes the same observations and rewards as the CPU DuelEnv from the same state."""
import copy

import jax
import jax.numpy as jnp
import numpy as np

from g1tt.duel_env import DuelEnv
from g1tt.env import EnvCfg
from g1tt.gpu.duel_jax import TOT_KEYS, GpuCfg, GpuDuel


def _cfg():
  return EnvCfg(level=1.0, obs_noise=0.0, sgs=None)


def _copy_state(cpu, gpu, S0):
  n = cpu.n
  S = dict(S0)
  st = lambda f: jnp.asarray(np.stack([getattr(x, f) for x in cpu.datas]), jnp.float32)
  S["d"] = S0["d"].replace(qpos=st("qpos"), qvel=st("qvel"), xmat=st("xmat").reshape(n, -1, 3, 3), xpos=st("xpos"),
                           site_xpos=st("site_xpos"), site_xmat=st("site_xmat").reshape(n, -1, 3, 3),
                           actuator_force=st("actuator_force"))
  sd = copy.deepcopy([vars(s) for s in cpu.sides])
  S["side"] = {f: jnp.asarray(np.stack([sd[0][f], sd[1][f]]).astype(np.asarray(S0["side"][f]).dtype))
               for f in S0["side"]}
  for f in ("t", "bp", "bv", "bw", "live", "to", "bounced", "hitter", "blade", "blade_v", "blade_n"):
    S[f] = jnp.asarray(np.asarray(getattr(cpu, f)).astype(np.asarray(S0[f]).dtype))
  return S


def test_observations_and_rewards_match_cpu():
  n = 4
  cpu = DuelEnv(2 * n, seed=1, cfg=_cfg())
  gpu = GpuDuel(GpuCfg(n=n, env=_cfg(), impl="jax", plan_k=n))
  C = dict(gpu.C)
  C["prm"] = jnp.asarray(np.stack([cpu.prm.kd, cpu.prm.km, cpu.prm.table_e, cpu.prm.table_mu, cpu.prm.rubber_e,
                                   cpu.prm.rubber_mu]), jnp.float32)
  S0 = gpu.init(jax.random.PRNGKey(0))
  obs_fn = jax.jit(gpu.observe)
  rew_fn = jax.jit(lambda C, S, ev, fell2: gpu._rewards(C, S, ev, fell2, {k: jnp.zeros(()) for k in TOT_KEYS}))
  rng = np.random.default_rng(0)
  checked, planned = 0, 0
  for k in range(90):
    fell2, timeout = cpu._advance(rng.normal(0, 0.3, (2 * n, 29)))
    if k % 30 == 29:
      S = _copy_state(cpu, gpu, S0)
      ev = {kk: jnp.asarray(np.asarray(vv).astype(np.float32 if np.asarray(vv).dtype.kind == "f" else bool))
            for kk, vv in cpu.ev.items()}
      rew_cpu, _ = cpu._rewards(fell2)
      S2, rew_gpu, _, _ = rew_fn(C, S, ev, jnp.asarray(fell2))
      obs_cpu = super(DuelEnv, cpu).observe()
      cobs_cpu = cpu.critic_obs(obs_cpu)
      _, obs_gpu, cobs_gpu = obs_fn(C, S2)
      np.testing.assert_allclose(np.asarray(rew_gpu), rew_cpu, atol=1e-4)
      np.testing.assert_allclose(np.asarray(obs_gpu), obs_cpu, atol=1e-4)
      np.testing.assert_allclose(np.asarray(cobs_gpu), cobs_cpu, atol=1e-4)
      checked += 1
      planned += int(sum(s.plan_valid.sum() for s in cpu.sides))
    else:
      cpu._rewards(fell2)
      super(DuelEnv, cpu).observe()
    cpu._finish(fell2, timeout)
  assert checked == 3
  assert planned > 0  # the states compared had planned strokes
