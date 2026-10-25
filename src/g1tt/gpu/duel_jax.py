"""The self-play duel as one compiled JAX step over N matches (2N robots).

Mirrors g1tt.duel.Duel + g1tt.duel_env.DuelEnv: same observations (actor and
critic layouts), rewards, rally rules, human reference strokes, Success Guided
Sampling and domain randomisation, so policies and checkpoints move between
the two. Physics goes through MJX: impl "warp" (MuJoCo Warp) on an NVIDIA GPU,
"jax" for tests on a laptop.

Differences from the CPU environment, all small:
- planning runs at the end of a control step (the CPU plans the moment the
  ball is hit, up to 20 ms earlier); at most `plan_k` balls are planned per
  step and any others wait for the next one;
- time restarts at 0 when a match resets, so float32 keeps millisecond
  resolution through days of training;
- SGS outcomes recorded for the same configuration in the same tick overwrite
  one ring slot instead of filling two.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from g1tt import planner
from g1tt.ball import KD, KM, BallParams
from g1tt.ball_jit import _first_crossing, _shortfall
from g1tt.duel import Duel
from g1tt.env import BALL_TICKS, CRITIC_DIM, CTRL_DT, REF_LOOKAHEAD, SUBSTEPS, EnvCfg, _action_scale, scaled_launch
from g1tt.gpu.kernels import RefTables, paddle_contact, step_world
from g1tt.model import BLADE_RADIUS, OPP_PREFIX, ROBOT_X, TABLE_H, TABLE_LEN, TABLE_WID, default_joint_targets
from g1tt.plan_batch import _plan
from g1tt.reference import IMITATE, LEGS, UPPER, RefLib

F32 = jnp.float32
FLIP = jnp.array([-1.0, -1.0, 1.0], F32)
WRIST = np.array([26, 27, 28])
HIDDEN = jnp.array([0.0, 0.0, -5.0], F32)


def flip(x, s):
  """World <-> robot s's frame for (..., 3) vectors; s may be per match."""
  if isinstance(s, (int, np.integer)):
    return x * FLIP if s else x
  return x * jnp.where((jnp.asarray(s) == 1)[..., None], FLIP, jnp.ones(3, F32))


def _yaw(quat):
  w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot_yaw(yaw, x):
  """_rot(_yaw_rot(yaw), x): world vector into the heading frame."""
  c, s = jnp.cos(yaw), jnp.sin(yaw)
  return jnp.stack([c * x[:, 0] + s * x[:, 1], -s * x[:, 0] + c * x[:, 1], x[:, 2]], -1)


def gpu_duel_model() -> mujoco.MjModel:
  """The two-G1 duel model for MJX: sensors and display meshes removed (the duel
  reads no sensors, and mesh geoms never collide), masses and inertias kept."""
  from g1tt.model import build_duel_spec
  spec = build_duel_spec()
  for s in list(spec.sensors):
    spec.delete(s)
  full = spec.compile()
  for g in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
    spec.delete(g)
  for msh in list(spec.meshes):
    spec.delete(msh)
  m = spec.compile()
  assert np.allclose(m.body_mass, full.body_mass) and np.allclose(m.body_inertia, full.body_inertia), \
    "mesh geoms carried mass: keep them"
  return m


def single_robot_consts() -> dict:
  """Ready pose, joint ranges and ready blade pose of one G1 (needs the G1 assets)."""
  from g1tt.model import build_model, ready_qpos
  single = build_model()
  q0 = ready_qpos(single)
  ds = mujoco.MjData(single)
  ds.qpos[:] = q0
  mujoco.mj_forward(single, ds)
  sid = single.site("blade").id
  return dict(q0=q0, jnt_lo=single.jnt_range[1:, 0], jnt_hi=single.jnt_range[1:, 1],
              ready_blade=ds.site_xpos[sid] - q0[:3], ready_normal=ds.site_xmat[sid].reshape(3, 3)[:, 1].copy())


@dataclasses.dataclass(frozen=True)
class GpuCfg:
  n: int  # matches; 2n robots
  env: EnvCfg
  impl: str = "warp"
  plan_k: int = 0  # balls planned per step; 0: n // 4
  seed: int = 0


class GpuDuel:
  """Host-side setup (model, constants, serve bank) and the compiled step."""

  def __init__(self, cfg: GpuCfg, model: mujoco.MjModel | None = None):
    self.cfg = cfg
    n = cfg.n
    e = cfg.env
    self.plan_k = cfg.plan_k or max(8, n // 4)
    self.rng = np.random.default_rng(cfg.seed)
    m = model if model is not None else gpu_duel_model()
    self.m = m
    self.mx = mjx.put_model(m, impl=cfg.impl)
    self.pelvis = np.array([m.body("pelvis").id, m.body(OPP_PREFIX + "pelvis").id])
    self.blade_sid = np.array([m.site("blade").id, m.site(OPP_PREFIX + "blade").id])
    self.max_steps = int(round(e.episode_s / CTRL_DT))
    self.phys_dt = float(m.opt.timestep)
    self.sgs = e.sgs
    self.bh_frac = e.bh_frac
    self.wide_frac = e.wide_frac
    pc = Duel.base_plan_cfg()
    self.base_plan_cfg = pc
    self.plan_cfg = dataclasses.replace(pc, land_y=e.land_y, hit_y=Duel.CHASE_Y, hit_z=Duel.CHASE_Z)
    self.reach_cfg = dataclasses.replace(pc, partner_y=(Duel.CHASE_Y[0] + 0.05, Duel.CHASE_Y[1] - 0.05),
                                         partner_z=Duel.CHASE_Z)
    self.ref_lib = RefLib()

    # Reset pose and the kinematics MJX would only have after a step.
    sc = single_robot_consts()
    q0 = sc["q0"]
    qb = q0.copy()
    qb[0:2] = -qb[0:2]
    qb[3:7] = [0.0, 0.0, 0.0, 1.0]
    d = mujoco.MjData(m)
    d.qpos[:36], d.qpos[36:] = q0, qb
    d.ctrl[:29] = d.ctrl[29:] = default_joint_targets()
    mujoco.mj_forward(m, d)
    lo, hi = sc["jnt_lo"], sc["jnt_hi"]
    self.d_reset = d

    C = dict(
      qpos0=d.qpos.copy(), ctrl0=d.ctrl.copy(), xmat0=d.xmat.reshape(-1, 3, 3).copy(), xpos0=d.xpos.copy(),
      site_xpos0=d.site_xpos.copy(), site_xmat0=d.site_xmat.reshape(-1, 3, 3).copy(),
      q_default=default_joint_targets(), act_scale=_action_scale(), jnt_lo=lo, jnt_hi=hi,
      ready_blade=sc["ready_blade"], ready_normal=sc["ready_normal"],
    )
    self.C = {k: jnp.asarray(v, F32) for k, v in C.items()}
    self.C["ref"] = RefTables.from_lib(self.ref_lib)
    self._c = self.C  # what methods read: the host constants, or step()'s traced argument
    prm = BallParams.randomised(self.rng, n, e.dr_scale)
    self.C["prm"] = jnp.asarray(np.stack([prm.kd, prm.km, prm.table_e, prm.table_mu, prm.rubber_e, prm.rubber_mu]), F32)
    self.set_level(e.level)

  # ---------------------------------------------------------------- host side
  def set_level(self, level: float):
    """Serve bank for this difficulty, padded to 2048 rows (count valid)."""
    self.cfg.env.level = level
    nominal = BallParams.nominal(2048)
    p0, v0, w0, pl, ok = planner.serve_and_plan(self.rng, 2048, scaled_launch(self.cfg.env.launch, level),
                                                planner.PlanCfg(), nominal, nominal)
    keep = np.flatnonzero(ok)
    pad = lambda x: np.concatenate([x[keep], np.zeros((2048 - len(keep),) + x.shape[1:], x.dtype)])
    bank = dict(p0=pad(p0), v0=pad(v0), w0=pad(w0))
    for f in pl.__dataclass_fields__:
      bank["pl_" + f] = pad(getattr(pl, f))
    self.C["bank"] = {k: jnp.asarray(v, jnp.bool_ if v.dtype == bool else F32) for k, v in bank.items()}
    self.C["bank_n"] = jnp.asarray(len(keep), jnp.int32)
    self.bank_reset = True  # SGS serve estimates start afresh

  def init(self, key):
    """Initial state: robots standing, no ball, episode clocks spread out."""
    n, e = self.cfg.n, self.cfg.env
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # One world's data, then per-world fields broadcast to n worlds. Under Warp
    # the contact buffers are shared by all worlds (naconmax is a total), so
    # they stay unbatched: vmapping make_data would copy them into every world.
    d1 = mjx.make_data(self.m, impl=self.cfg.impl, naconmax=48 * n, njmax=256)
    d1 = d1.replace(qpos=jnp.asarray(self.d_reset.qpos, F32), ctrl=jnp.asarray(self.d_reset.ctrl, F32))
    self.data_axes = self._batch_axes(d1)
    d = jax.tree_util.tree_map(lambda ax, x: jnp.broadcast_to(x, (n,) + x.shape) if ax == 0 else x,
                               self.data_axes, d1, is_leaf=lambda x: x is None)
    d = self._reset_data(d, jnp.ones(n, bool))
    z = lambda *s: jnp.zeros(s, F32)
    side = dict(
      plan_valid=jnp.zeros((2, n), bool), t_hit=jnp.full((2, n), jnp.inf, F32), p_star=z(2, n, 3), n_star=z(2, n, 3),
      v_star=z(2, n, 3), hit_done=jnp.zeros((2, n), bool), ref_on=jnp.zeros((2, n), bool),
      ref_clip=jnp.zeros((2, n), jnp.int32), ref_stretch=jnp.ones((2, n), F32), ref_timp=jnp.full((2, n), jnp.inf, F32),
      ref_shift=z(2, n, 3), last_action=z(2, n, 29), prev_action=z(2, n, 29), vout_star=z(2, n, 3),
      land_tgt=z(2, n, 3), face=jnp.ones((2, n), F32), wide=jnp.zeros((2, n), bool),
      aim=jnp.zeros((2, n), jnp.int32), cell=jnp.full((2, n), -1, jnp.int32))
    blade = jnp.stack([jnp.broadcast_to(self._c["site_xpos0"][self.blade_sid[s]], (n, 3)) for s in range(2)])
    blade_n = jnp.stack([jnp.broadcast_to(self._c["site_xmat0"][self.blade_sid[s]].reshape(3, 3)[:, 1], (n, 3))
                         for s in range(2)])
    S = dict(
      d=d, side=side, t=z(n), steps=jax.random.randint(k1, (n,), 0, self.max_steps) if e.desync else jnp.zeros(n, jnp.int32),
      bp=jnp.broadcast_to(HIDDEN, (n, 3)), bv=z(n, 3), bw=z(n, 3), live=jnp.zeros(n, bool),
      to=jnp.zeros(n, jnp.int32), bounced=jnp.zeros(n, bool), hitter=jnp.full(n, -1, jnp.int32),
      next_receiver=jax.random.randint(k2, (n,), 0, 2), timer=jax.random.uniform(k3, (n,), F32, 0.6, 1.2),
      ball_t=z(n), rally_len=jnp.zeros(n, jnp.int32), last_face=z(n), replan_due=jnp.zeros(n, bool),
      need_plan=jnp.zeros(n, bool),  # a ball was hit: plan the receiver's return (bounces_left 1)
      blade=blade, blade_v=z(2, n, 3), blade_n=blade_n,
      hit_p=z(n, 3), hit_v=z(n, 3), hit_w=z(n, 3),  # ball just after its last hit, hitter's frame
    )
    if self.sgs is not None:
      H = int(self.sgs.get("history", 100))
      for name, size in (("serve", 2048), ("ret", 4 * Duel.SGS_BINS ** 2)):
        S[f"sgs_{name}_out"] = jnp.zeros((size, H), jnp.int8)
        S[f"sgs_{name}_count"] = jnp.zeros(size, jnp.int32)
        S[f"sgs_{name}_next"] = jnp.zeros(size, jnp.int32)
      for k in ("recv_kind", "hit_kind"):
        S[k] = jnp.full(n, -1, jnp.int32)
      for k in ("recv_idx", "hit_idx"):
        S[k] = jnp.zeros(n, jnp.int32)
      S["recv_arrived"] = jnp.zeros(n, bool)
    return S

  def reset_sgs_serve(self, S):
    if self.sgs is not None:
      S = dict(S)
      S["sgs_serve_out"] = jnp.zeros_like(S["sgs_serve_out"])
      S["sgs_serve_count"] = jnp.zeros_like(S["sgs_serve_count"])
      S["sgs_serve_next"] = jnp.zeros_like(S["sgs_serve_next"])
    return S

  # ---------------------------------------------------------------- physics state
  def _batch_axes(self, d1):
    """vmap axis of each mjx.Data leaf: 0 per world, None shared (Warp's contact buffers)."""
    if self.cfg.impl != "warp":
      return jax.tree_util.tree_map(lambda x: 0, d1)
    from mujoco.mjx.warp import types as wt
    batched = wt._BATCH_DIM["Data"]

    def axis(path, x):
      name = getattr(path[-1], "name", getattr(path[-1], "key", str(path[-1])))
      return 0 if batched.get(name, True) else None
    return jax.tree_util.tree_map_with_path(axis, d1)

  def _reset_data(self, d, mask):
    """Robots back to the ready pose in matches `mask`, kinematics included."""
    C = self.C
    w = lambda x, v: jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), v, x)
    d = d.replace(qpos=w(d.qpos, C["qpos0"]), qvel=w(d.qvel, 0.0), ctrl=w(d.ctrl, C["ctrl0"]),
                  qacc_warmstart=w(d.qacc_warmstart, 0.0), xmat=w(d.xmat, C["xmat0"]), xpos=w(d.xpos, C["xpos0"]),
                  site_xpos=w(d.site_xpos, C["site_xpos0"]), site_xmat=w(d.site_xmat, C["site_xmat0"]))
    return d

  def _state(self, S, s):
    """Base, yaw, gyro, gravity, joints, joint velocities of robot s, in its frame."""
    d = S["d"]
    q = d.qpos[:, 36 * s:36 * s + 36]
    qv = d.qvel[:, 35 * s:35 * s + 35]
    base = flip(q[:, 0:3], s)
    yaw = _yaw(q[:, 3:7]) + (jnp.pi if s else 0.0)
    R = d.xmat[:, self.pelvis[s]].reshape(-1, 3, 3)
    return base, yaw, qv[:, 3:6], -R[:, 2, :], q[:, 7:], qv[:, 6:]

  def _read_blades(self, d):
    blade = jnp.stack([d.site_xpos[:, self.blade_sid[s]] for s in range(2)])
    blade_n = jnp.stack([d.site_xmat[:, self.blade_sid[s]].reshape(-1, 3, 3)[:, :, 1] for s in range(2)])
    return blade, blade_n

  # ---------------------------------------------------------------- sides
  @staticmethod
  def _clear(side, mask):
    """Side.clear for matches mask (2, n) or (n,) applied to both sides."""
    m = jnp.broadcast_to(mask, side["plan_valid"].shape)
    side = dict(side)
    side["plan_valid"] = side["plan_valid"] & ~m
    side["t_hit"] = jnp.where(m, jnp.inf, side["t_hit"])
    side["hit_done"] = side["hit_done"] & ~m
    side["ref_on"] = side["ref_on"] & ~m
    side["ref_timp"] = jnp.where(m, jnp.inf, side["ref_timp"])
    return side

  def _set_plan(self, side, s, mask, pl, t_now, keep_stroke, key):
    """Side.set_plan for side s on matches `mask`; pl holds full-length (n) plan arrays."""
    ref = self._c["ref"]
    get = lambda k: side[k][s]
    v = mask & pl["valid"]
    upd = {}
    upd["plan_valid"] = jnp.where(mask, pl["valid"], get("plan_valid"))
    vv = v[:, None]
    t_hit = jnp.where(v, t_now + pl["t_hit"], get("t_hit"))
    p_star = jnp.where(vv, pl["blade_pos"], get("p_star"))
    v_star = jnp.where(vv, pl["blade_vel"], get("v_star"))
    upd.update(t_hit=t_hit, p_star=p_star, v_star=v_star,
               vout_star=jnp.where(vv, pl["ball_vel_out"], get("vout_star")),
               land_tgt=jnp.where(vv, pl["land"], get("land_tgt")))
    new = v & (~get("ref_on") if keep_stroke else True)
    clip, stretch = ref.choose(key, p_star - jnp.array([ROBOT_X, 0.0, 0.0], F32), v_star)
    ref_clip = jnp.where(new, clip, get("ref_clip"))
    upd["ref_clip"] = ref_clip
    upd["ref_stretch"] = jnp.where(new, stretch, get("ref_stretch"))
    ref_on = get("ref_on") | new
    upd["ref_on"] = ref_on
    face = jnp.where(new, jnp.where(ref.cls[clip] == 4, -1.0, 1.0), get("face"))
    upd["face"] = face
    upd["n_star"] = jnp.where(vv, pl["normal"] * face[:, None], get("n_star"))
    upd["ref_timp"] = jnp.where(v, t_hit, get("ref_timp"))
    on = v & ref_on
    shift = p_star - jnp.array([ROBOT_X, 0.0, 0.0], F32) - ref.imp_blade[ref_clip]
    upd["ref_shift"] = jnp.where(on[:, None], shift, get("ref_shift"))
    side = dict(side)
    for k, x in upd.items():
      side[k] = side[k].at[s].set(x)
    return side

  def _window(self, side, s, t, ahead):
    """Side._window: (active, rel) and ref_on switched off past the stroke's end."""
    ref = self._c["ref"]
    rel = t + ahead - side["ref_timp"][s]
    pre, post = ref.window(side["ref_clip"][s], side["ref_stretch"][s])
    on = side["ref_on"][s]
    active = on & (rel >= -pre) & (rel <= post)
    side = dict(side)
    side["ref_on"] = side["ref_on"].at[s].set(on & ~(rel > post))
    return side, active, rel

  # ---------------------------------------------------------------- aims and SGS
  def _choose_aims(self, S, s_of, mask, key):
    """Duel._choose_aims for the robots s_of[i] of matches mask."""
    n = self.cfg.n
    k1, k2, k3, k4 = jax.random.split(key, 4)
    r = jax.random.uniform(k1, (n,))
    aim = jnp.where(r < self.wide_frac, 1, jnp.where(r < self.wide_frac + self.bh_frac, 3, 0)).astype(jnp.int32)
    cell = jnp.full(n, -1, jnp.int32)
    if self.sgs is not None:
      b2 = Duel.SGS_BINS ** 2
      cell = aim * b2 + jax.random.randint(k2, (n,), 0, b2)
      by = jax.random.uniform(k3, (n,)) < self.sgs.get("mix", 0.5)
      cell = jnp.where(by, self._sgs_sample(S, "ret", k4, n), cell)
      aim = cell // b2
    side = dict(S["side"])
    idx = jnp.arange(n)
    side["aim"] = side["aim"].at[s_of, idx].set(jnp.where(mask, aim, side["aim"][s_of, idx]))
    side["cell"] = side["cell"].at[s_of, idx].set(jnp.where(mask, cell, side["cell"][s_of, idx]))
    S = dict(S)
    S["side"] = side
    return S

  def _sgs_probs(self, S, name, valid=None):
    g = self.sgs
    t, kap, eps, T = g.get("target", 0.5), g.get("kappa", 5.0), 1e-4, max(g.get("temperature", 2.0), 1.0)
    H = S[f"sgs_{name}_out"].shape[1]
    seen = jnp.minimum(S[f"sgs_{name}_count"], H)
    p = jnp.where(seen > 0, S[f"sgs_{name}_out"].sum(1) / jnp.maximum(seen, 1), t)
    w = (p + eps) ** (kap * t) * (1 - p + eps) ** (kap * (1 - t))
    logit = jnp.log(jnp.maximum(w, eps) + eps) / T
    if valid is not None:
      logit = jnp.where(valid, logit, -jnp.inf)
    return jax.nn.softmax(logit), p, seen

  def _sgs_sample(self, S, name, key, k):
    valid = jnp.arange(2048) < self._c["bank_n"] if name == "serve" else None
    probs, _, _ = self._sgs_probs(S, name, valid)
    return jax.random.choice(key, probs.shape[0], (k,), p=probs).astype(jnp.int32)

  def _sgs_record(self, S, mask, success):
    """SGS outcome for the configuration each match `mask` just played (hit_*)."""
    S = dict(S)
    for kind, name in ((1, "serve"), (2, "ret")):
      m = mask & (S["hit_kind"] == kind)
      idx = jnp.where(m, S["hit_idx"], 0)
      out, cnt, nxt = S[f"sgs_{name}_out"], S[f"sgs_{name}_count"], S[f"sgs_{name}_next"]
      H = out.shape[1]
      pos = nxt[idx]
      row = jnp.where(m, idx, out.shape[0])  # out-of-range rows are dropped
      out = out.at[row, pos].set(jnp.int8(1 if success else 0), mode="drop")
      cnt = cnt.at[row].add(1, mode="drop")
      nxt = nxt.at[row].set((pos + 1) % H, mode="drop")
      S[f"sgs_{name}_out"], S[f"sgs_{name}_count"], S[f"sgs_{name}_next"] = out, cnt, nxt
    S["hit_kind"] = jnp.where(mask, -1, S["hit_kind"])
    return S

  # ---------------------------------------------------------------- serve, rally end
  def _serve(self, S, mask, key):
    C, n = self._c, self.cfg.n
    b = C["bank"]
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    j = jax.random.randint(k1, (n,), 0, C["bank_n"])
    if self.sgs is not None:
      by = jax.random.uniform(k2, (n,)) < self.sgs.get("mix", 0.5)
      j = jnp.where(by, self._sgs_sample(S, "serve", k3, n), j)
    recv = S["next_receiver"]
    S = dict(S)
    mm = mask[:, None]
    S["bp"] = jnp.where(mm, flip(b["p0"][j], recv), S["bp"])
    S["bv"] = jnp.where(mm, flip(b["v0"][j], recv), S["bv"])
    S["bw"] = jnp.where(mm, flip(b["w0"][j], recv), S["bw"])
    S["live"] = S["live"] | mask
    S["to"] = jnp.where(mask, recv, S["to"])
    S["bounced"] = S["bounced"] & ~mask
    S["hitter"] = jnp.where(mask, -1, S["hitter"])
    S["ball_t"] = jnp.where(mask, 0.0, S["ball_t"])
    S["next_receiver"] = jnp.where(mask, 1 - recv, recv)
    if self.sgs is not None:
      S["recv_kind"] = jnp.where(mask, 1, S["recv_kind"])
      S["recv_idx"] = jnp.where(mask, j, S["recv_idx"])
      S["recv_arrived"] = S["recv_arrived"] & ~mask
      S["hit_kind"] = jnp.where(mask, -1, S["hit_kind"])
    side = self._clear(S["side"], mask)
    pl = {f: b["pl_" + f][j] for f in planner.Plan.__dataclass_fields__}
    for s in range(2):
      side = self._set_plan(side, s, mask & (recv == s), pl, S["t"], False, (k4, k5)[s])
    S["side"] = side
    if self.sgs is not None:
      S = self._choose_aims(S, recv, mask, jax.random.fold_in(key, 7))
    return S, mask.sum()

  def _end_rally(self, S, mask, reason, timer_new, tot):
    S = dict(S)
    tot = dict(tot)
    tot["rallies"] = tot["rallies"] + mask.sum()
    tot["rally_sum"] = tot["rally_sum"] + jnp.where(mask, S["rally_len"], 0).sum()
    tot["rally_max"] = jnp.maximum(tot["rally_max"], jnp.where(mask, S["rally_len"], 0).max())
    if self.sgs is not None and reason != "time":
      S = self._sgs_record(S, mask & (S["hit_kind"] >= 0), False)
      faced = mask & (S["recv_kind"] >= 0) & S["recv_arrived"]
      S["hit_kind"] = jnp.where(faced, S["recv_kind"], S["hit_kind"])
      S["hit_idx"] = jnp.where(faced, S["recv_idx"], S["hit_idx"])
      S = self._sgs_record(S, faced, False)
    if self.sgs is not None:
      S["hit_kind"] = jnp.where(mask, -1, S["hit_kind"])
      S["recv_kind"] = jnp.where(mask, -1, S["recv_kind"])
      S["recv_arrived"] = S["recv_arrived"] & ~mask
    S["rally_len"] = jnp.where(mask, 0, S["rally_len"])
    S["live"] = S["live"] & ~mask
    S["bp"] = jnp.where(mask[:, None], HIDDEN, S["bp"])
    S["timer"] = jnp.where(mask, timer_new, S["timer"])
    S["side"] = self._clear(S["side"], mask)
    S["need_plan"] = S["need_plan"] & ~mask
    return S, tot

  # ---------------------------------------------------------------- planning
  def _plan(self, S, mask, bounces_left, key):
    """Duel._plan_for (batched planner) for robot `to` of matches `mask`;
    up to plan_k at a time, the rest stay in `mask` for the next step."""
    n, K = self.cfg.n, self.plan_k
    C = self._c
    s_of = S["to"]
    side = S["side"]
    if bounces_left == 0:  # after the bounce a valid plan is only refined
      mask = mask & side["plan_valid"][s_of, jnp.arange(n)]
    k_aim, k_plan, k_set0, k_set1 = jax.random.split(key, 4)
    if bounces_left == 1:
      S = self._choose_aims(S, s_of, mask, k_aim)
      side = S["side"]
    count = mask.sum()
    idx = jnp.nonzero(mask, size=K, fill_value=0)[0]
    slot = jnp.arange(K) < count
    done = jnp.zeros(n, bool).at[idx].set(slot)  # planned this step
    s_k = s_of[idx]
    f = lambda x: flip(x[idx], s_k)
    aim = side["aim"][s_k, idx]
    cell = side["cell"][s_k, idx]
    cfg = self.plan_cfg
    land_x = jnp.broadcast_to(jnp.array(cfg.land_x, F32), (K, 2))
    land_y = jnp.broadcast_to(jnp.array(cfg.land_y, F32), (K, 2))
    py = jnp.broadcast_to(jnp.array(cfg.partner_y, F32), (K, 2))
    retries = jnp.full(K, cfg.land_retries, jnp.int32)
    for a in (1, 2):
      m = (aim == a)[:, None]
      land_x = jnp.where(m, jnp.array((0.30, 0.90), F32), land_x)
      land_y = jnp.where(m, jnp.array(Duel.WIDE_LAND_Y[a - 1], F32), land_y)
      py = jnp.where(m, jnp.array(Duel.WIDE_Y[a - 1], F32), py)
      retries = jnp.where(aim == a, 6, retries)
    for a, ly, pyy in ((0, Duel.FH_LAND_Y, Duel.FH_Y), (3, Duel.BH_LAND_Y, Duel.BH_Y)):
      m = (aim == a)[:, None]
      land_y = jnp.where(m, jnp.array(ly, F32), land_y)
      py = jnp.where(m, jnp.array(pyy, F32), py)
      retries = jnp.where(aim == a, Duel.AIM_RETRIES, retries)
    if self.sgs is not None:
      k = Duel.SGS_BINS
      cc = jnp.maximum(cell, 0)
      for name, i in (("x", (cc % k ** 2) // k), ("y", cc % k)):
        r = land_x if name == "x" else land_y
        lo, hi = r[:, 0], r[:, 1]
        nr = jnp.stack([lo + (hi - lo) * i / k, lo + (hi - lo) * (i + 1) / k], -1)
        nr = jnp.where((cell >= 0)[:, None], nr, r)
        if name == "x":
          land_x = nr
        else:
          land_y = nr
    nom = lambda x: jnp.full(K, x, F32)
    horizon = 0.6 if bounces_left == 0 else 1.6
    steps = int(round(horizon / 0.004))
    out = _plan(k_plan, f(S["bp"]), f(S["bv"]), f(S["bw"]), nom(KD), nom(KM), nom(0.88), nom(0.25), nom(0.83), nom(0.6),
                land_x, land_y, py, retries, nom(cfg.net_clearance), True, Duel.BH_CLEAR,
                jnp.broadcast_to(jnp.array(Duel.BH_LAND_X, F32), (K, 2)), jnp.zeros((K, 3), F32),
                cfg.hit_x, cfg.hit_y[0], cfg.hit_y[1], cfg.hit_z[0], cfg.hit_z[1], cfg.partner_z[0], cfg.partner_z[1],
                self.ref_lib.BACKHAND_FROM_Y, horizon / steps, jnp.arange(steps),
                bounces=bounces_left, strict=bounces_left == 1, ts=tuple(float(t) for t in cfg.return_t),
                prefer=float(cfg.prefer_out_speed), band=True, reach_steps=int(round(1.3 / 0.006)),
                max_retries=max(6, Duel.AIM_RETRIES), fixed_land=False)
    names = list(planner.Plan.__dataclass_fields__)
    full = {}
    for name, x in zip(names, out):
      base = jnp.zeros((n,) + x.shape[1:], x.dtype)
      full[name] = base.at[idx].set(jnp.where(slot.reshape((-1,) + (1,) * (x.ndim - 1)), x, base[idx]))
    S = dict(S)
    tot_unreach = 0
    if bounces_left == 1:
      b = self.plan_cfg
      bp = full["ball_pos"]
      base_cfg = self.base_plan_cfg
      in_band = ((bp[:, 2] > base_cfg.hit_z[0]) & (bp[:, 2] < base_cfg.hit_z[1])
                 & (bp[:, 1] > base_cfg.hit_y[0]) & (bp[:, 1] < base_cfg.hit_y[1]))
      wide = full["valid"] & ~in_band
      side = dict(S["side"])
      side["wide"] = side["wide"].at[s_of, jnp.arange(n)].set(jnp.where(done, wide, side["wide"][s_of, jnp.arange(n)]))
      S["side"] = side
      tot_unreach = (done & ~full["valid"]).sum()
    keep = done & full["valid"] if bounces_left == 0 else done
    side = S["side"]
    for s, ks in ((0, k_set0), (1, k_set1)):
      side = self._set_plan(side, s, keep & (s_of == s), full, S["t"], bounces_left == 0, ks)
    S["side"] = side
    return S, mask & ~done, tot_unreach

  # ---------------------------------------------------------------- one control step
  def step(self, C, S, actions, key):
    """actions (2n, 29) -> new state and what training needs (see DuelEnv.step)."""
    n, e, rc = self.cfg.n, self.cfg.env, self.cfg.env.reward
    self._c = C
    keys = jax.random.split(key, 8)
    tot = {k: jnp.zeros((), F32) for k in TOT_KEYS}
    S, ev, fell2, timeout, tot = self._advance(C, S, actions, keys[0], tot)
    S, rew, terms, tot = self._rewards(C, S, ev, fell2, tot)
    S, obs_pre = self._observe(C, S)
    term_cobs = self._critic_obs(C, S, obs_pre)
    match_done = fell2.any(0) | timeout
    fell = jnp.concatenate([fell2[0], fell2[1]])
    cut = jnp.concatenate([match_done, match_done]) & ~fell
    tot["falls"] = tot["falls"] + fell.sum()
    tot["episodes"] = tot["episodes"] + (fell | cut).sum()
    S, tot = self._finish(C, S, fell2, timeout, keys[1], tot)
    S, obs_clean = self._observe(C, S)
    obs = obs_clean + e.obs_noise * jax.random.normal(keys[2], obs_clean.shape, F32) if e.obs_noise else obs_clean
    cobs = self._critic_obs(C, S, obs_clean)
    return S, dict(obs=obs, cobs=cobs, rew=rew, fell=fell, cut=cut, term_cobs=term_cobs, terms=terms, tot=tot)

  def observe(self, C, S):
    self._c = C
    S, o = self._observe(C, S)
    return S, o, self._critic_obs(C, S, o)

  def _advance(self, C, S, actions, key, tot):
    n = self.cfg.n
    dt = self.phys_dt
    h = dt / BALL_TICKS
    prm = C["prm"]
    kd, km, te, tmu, re, rmu = (prm[i] for i in range(6))
    a = jnp.clip(actions, -4.0, 4.0).reshape(2, n, 29)
    S = dict(S)
    side = dict(S["side"])
    side["prev_action"] = side["last_action"]
    side["last_action"] = a
    S["side"] = side
    targets = jnp.clip(C["q_default"] + C["act_scale"] * a, C["jnt_lo"], C["jnt_hi"])
    k_serve, k_timer, k_plan1, k_plan0 = jax.random.split(key, 4)
    S, served = self._serve(S, ~S["live"] & (S["timer"] <= 0), k_serve)
    tot["serves"] = tot["serves"] + served
    timer_new = jax.random.uniform(k_timer, (n,), F32, 0.6, 1.2)
    ctrl = jnp.concatenate([targets[0], targets[1]], -1)
    z = lambda *s: jnp.zeros((2, n) + s, F32)
    ev = dict(hit=jnp.zeros((2, n), bool), vout=z(3), land=jnp.zeros((2, n), bool), land_err=z(),
              miss=jnp.zeros((2, n), bool), fault=jnp.zeros((2, n), bool), imp=jnp.zeros((2, n), bool),
              imp_pos=z(), imp_vel=z(), imp_cos=z(), hit_face_ok=jnp.zeros((2, n), bool),
              land_face_ok=jnp.zeros((2, n), bool))
    # Axis 0 for the whole Data: under Warp, DataWarp is a registered vmappable
    # that itself leaves the shared contact buffers unbatched.
    step_fn = jax.vmap(lambda d: mjx.step(self.mx, d))
    idx = jnp.arange(n)

    def substep(carry, _):
      S, ev, tot = carry
      S = dict(S)
      c0, n0 = S["blade"], S["blade_n"]
      d = step_fn(S["d"].replace(ctrl=ctrl))
      S["d"] = d
      blade, blade_n = self._read_blades(d)
      vpad = (blade - c0) / dt
      wpad = jnp.cross(n0, blade_n) / dt
      t_prev = S["t"]
      S["t"] = S["t"] + dt
      S["timer"] = S["timer"] - dt
      ev = dict(ev)
      side = S["side"]
      for s in range(2):
        cross = side["plan_valid"][s] & (t_prev < side["t_hit"][s]) & (S["t"] >= side["t_hit"][s])
        fb, fv, fn = flip(blade[s], s), flip(vpad[s], s), flip(blade_n[s], s)
        ev["imp"] = ev["imp"].at[s].set(ev["imp"][s] | cross)
        ev["imp_pos"] = ev["imp_pos"].at[s].set(jnp.where(cross, jnp.linalg.norm(fb - side["p_star"][s], axis=-1), ev["imp_pos"][s]))
        ev["imp_vel"] = ev["imp_vel"].at[s].set(jnp.where(cross, jnp.linalg.norm(fv - side["v_star"][s], axis=-1), ev["imp_vel"][s]))
        ev["imp_cos"] = ev["imp_cos"].at[s].set(jnp.where(cross, jnp.sum(fn * side["n_star"][s], -1), ev["imp_cos"][s]))
      S["blade"], S["blade_n"], S["blade_v"] = blade, blade_n, vpad
      for k in range(BALL_TICKS):
        a0, a1 = k / BALL_TICKS, (k + 1) / BALL_TICKS
        to = S["to"]
        C0 = c0[to, idx] + a0 * (blade[to, idx] - c0[to, idx])
        C1 = c0[to, idx] + a1 * (blade[to, idx] - c0[to, idx])
        N0 = n0[to, idx] + a0 * (blade_n[to, idx] - n0[to, idx])
        N1 = n0[to, idx] + a1 * (blade_n[to, idx] - n0[to, idx])
        p_prev = S["bp"]
        live = S["live"]
        p, v, w, table, net, floor = step_world(S["bp"], S["bv"], S["bw"], h, kd, km, te, tmu, live)
        side = S["side"]
        hit_done_to = jnp.where(to == 0, side["hit_done"][0], side["hit_done"][1])
        can_hit = live & S["bounced"] & ~hit_done_to
        p, v, w, hit, face_side = paddle_contact(p_prev, p, v, w, C0, C1, N0, N1, vpad[to, idx], wpad[to, idx],
                                                 BLADE_RADIUS, can_hit, re, rmu)
        sd_face = jnp.where(to == 0, side["face"][0], side["face"][1])
        tot["cmd_bh"] = tot["cmd_bh"] + (hit & (sd_face < 0)).sum()
        tot["cmd_bh_on_bh"] = tot["cmd_bh_on_bh"] + (hit & (sd_face < 0) & (face_side < 0)).sum()
        S["last_face"] = jnp.where(hit, face_side, S["last_face"])
        S["bp"], S["bv"], S["bw"] = p, v, w
        S["ball_t"] = S["ball_t"] + h * live
        half = (p[:, 0] > 0).astype(jnp.int32)
        on_to = table & (half == to)
        landed = on_to & ~S["bounced"] & (S["hitter"] >= 0)
        S["rally_len"] = S["rally_len"] + landed
        if self.sgs is not None:
          S = self._sgs_record(S, landed & (S["hit_kind"] >= 0), True)
        for s in range(2):
          ls = landed & (S["hitter"] == s)
          ev["land"] = ev["land"].at[s].set(ev["land"][s] | ls)
          ev["land_face_ok"] = ev["land_face_ok"].at[s].set(
            ev["land_face_ok"][s] | (ls & (S["last_face"] * S["side"]["face"][s] > 0)))
          pf = flip(p, s)
          ev["land_err"] = ev["land_err"].at[s].set(
            jnp.where(ls, jnp.linalg.norm(pf[:, :2] - S["side"]["land_tgt"][s][:, :2], axis=-1), ev["land_err"][s]))
        first_bounce = on_to & ~S["bounced"]
        S["bounced"] = S["bounced"] | first_bounce
        S["replan_due"] = S["replan_due"] | first_bounce
        if self.sgs is not None:
          S["recv_arrived"] = S["recv_arrived"] | first_bounce
        bounced, hitter = S["bounced"], S["hitter"]
        fault = live & ~bounced & (hitter >= 0) & ((table & (half != to)) | net | floor)
        void = live & ~bounced & (hitter < 0) & ((table & (half != to)) | net | floor)
        behind = jnp.where(to == 0, p[:, 0] < ROBOT_X - 0.6, p[:, 0] > -ROBOT_X + 0.6)
        miss = live & bounced & ~hit & ((on_to & ~first_bounce) | (table & (half != to)) | net | floor
                                        | behind | (S["ball_t"] > 3.0))
        for s in range(2):
          ev["fault"] = ev["fault"].at[s].set(ev["fault"][s] | (fault & (hitter == s)))
          ev["miss"] = ev["miss"].at[s].set(ev["miss"][s] | (miss & (to == s)))
        for reason, m in (("fault", fault), ("miss", miss), ("fault", void)):
          S, tot = self._end_rally(S, m, reason, timer_new, tot)
        good = hit & S["live"]
        side = dict(S["side"])
        for s in range(2):
          mine = good & (to == s)
          fs = (-1.0 if s else 1.0)
          ev["hit"] = ev["hit"].at[s].set(ev["hit"][s] | mine)
          ev["hit_face_ok"] = ev["hit_face_ok"].at[s].set(
            jnp.where(mine, face_side * side["face"][s] > 0, ev["hit_face_ok"][s]))
          ev["vout"] = ev["vout"].at[s].set(jnp.where(mine[:, None], flip(v, s), ev["vout"][s]))
          side["hit_done"] = side["hit_done"].at[s].set(side["hit_done"][s] | mine)
        S["hit_p"] = jnp.where(good[:, None], flip(p, to), S["hit_p"])
        S["hit_v"] = jnp.where(good[:, None], flip(v, to), S["hit_v"])
        S["hit_w"] = jnp.where(good[:, None], flip(w, to), S["hit_w"])
        S["side"] = side
        if self.sgs is not None:
          S["hit_kind"] = jnp.where(good, S["recv_kind"], S["hit_kind"])
          S["hit_idx"] = jnp.where(good, S["recv_idx"], S["hit_idx"])
          cell = jnp.where(to == 0, side["cell"][0], side["cell"][1])
          S["recv_kind"] = jnp.where(good, jnp.where(cell >= 0, 2, -1), S["recv_kind"])
          S["recv_idx"] = jnp.where(good, jnp.maximum(cell, 0), S["recv_idx"])
          S["recv_arrived"] = S["recv_arrived"] & ~good
        S["hitter"] = jnp.where(good, to, S["hitter"])
        S["to"] = jnp.where(good, 1 - to, to)
        S["bounced"] = S["bounced"] & ~good
        S["ball_t"] = jnp.where(good, 0.0, S["ball_t"])
        # The robot to play next starts afresh; its return is planned after the step.
        nxt_clear = jnp.stack([good & (S["to"] == 0), good & (S["to"] == 1)])
        S["side"] = self._clear(S["side"], nxt_clear)
        S["need_plan"] = S["need_plan"] | good
      return (S, ev, tot), None

    (S, ev, tot), _ = jax.lax.scan(substep, (S, ev, tot), None, length=SUBSTEPS)

    # Shot quality at each hit (DuelEnv: shortfall and whether the partner can play it).
    hitter_side = S["hitter"]
    hit_any = ev["hit"].any(0)
    sf, pk = self._hit_quality(C, S, hit_any)
    ev["shortfall"] = jnp.stack([jnp.where(ev["hit"][s], sf, 0.0) for s in range(2)])
    ev["playable"] = jnp.stack([ev["hit"][s] & pk for s in range(2)])
    del hitter_side

    # Plans: returns of balls just hit, then refinements after the bounce.
    S, rest, unreach = self._plan(S, S["need_plan"], 1, k_plan1)
    S["need_plan"] = rest
    tot["unreachable"] = tot["unreachable"] + unreach
    due = S["replan_due"] & S["live"] & S["bounced"] & ~S["side"]["hit_done"][S["to"], idx]
    S, _, _ = self._plan(S, due, 0, k_plan0)
    S["replan_due"] = jnp.zeros(n, bool)

    S["steps"] = S["steps"] + 1
    fell = []
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(S, s)
      fell.append((base[:, 2] < 0.5) | (grav[:, 2] > -0.7))
    timeout = S["steps"] >= self.max_steps
    return S, ev, jnp.stack(fell), timeout, tot

  def _hit_quality(self, C, S, mask):
    """return_shortfall and partner reach for balls hit this step, compacted to plan_k."""
    n, K = self.cfg.n, self.plan_k
    prm = C["prm"]
    idx = jnp.nonzero(mask, size=K, fill_value=0)[0]
    slot = jnp.arange(K) < mask.sum()
    p, v, w = S["hit_p"][idx], S["hit_v"][idx], S["hit_w"][idx]
    kd, km, te, tmu = prm[0][idx], prm[1][idx], prm[2][idx], prm[3][idx]
    sf = _shortfall(p, v, w, kd, km, 0.004, jnp.zeros(int(1.5 / 0.004)))
    rc = self.reach_cfg
    found, pos = _first_crossing(p, v, w, kd, km, te, tmu, 0.006, -rc.hit_x, 1, jnp.zeros(int(round(2.0 / 0.006))))
    y = -pos[:, 1]
    pk = (found & (pos[:, 2] > rc.partner_z[0]) & (pos[:, 2] < rc.partner_z[1])
          & (y > rc.partner_y[0] + 0.05) & (y < rc.partner_y[1] - 0.05))
    sf_full = jnp.zeros(n, F32).at[idx].set(jnp.where(slot, sf, 0.0))
    pk_full = jnp.zeros(n, bool).at[idx].set(slot & pk)
    return sf_full, pk_full

  # ---------------------------------------------------------------- rewards (DuelEnv._rewards)
  def _rewards(self, C, S, ev, fell2, tot):
    rc, e, n = self.cfg.env.reward, self.cfg.env, self.cfg.n
    ref = C["ref"]
    rew, terms = [], {}
    side = S["side"]
    tot = dict(tot)
    d = S["d"]
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(S, s)
      blade = flip(S["blade"][s], s)
      bn = flip(S["blade_n"][s], s)
      bv = flip(S["blade_v"][s], s)
      torque = d.actuator_force[:, 29 * s:29 * s + 29]
      g = lambda k: side[k][s]
      tau = g("t_hit") - S["t"]
      my_turn = S["live"] & (S["to"] == s)
      active = g("plan_valid") & ~g("hit_done") & (tau > -0.06) & my_turn
      T = {}
      side, r_act, rel = self._window(side, s, S["t"], 0.0)
      r = jnp.where(r_act, rel, 0.0)
      pt = ref.blade_path(g("ref_clip"), r, g("ref_stretch"), g("ref_shift"))
      v_h = ref.imp_vel[g("ref_clip")] / g("ref_stretch")[:, None]
      pt = pt + (g("v_star") - v_h) * (r * jnp.exp(-(r / 0.15) ** 2))[:, None] + jnp.array([ROBOT_X, 0.0, 0.0], F32)
      T["path"] = rc.path * jnp.exp(-(jnp.linalg.norm(blade - pt, axis=-1) / 0.10) ** 2) * r_act
      q_ref = ref.joints(g("ref_clip"), r, g("ref_stretch"))
      imitate = r_act & (rel <= rc.ref_until)
      e_up = jnp.mean((qj[:, IMITATE] - q_ref[:, IMITATE]) ** 2, -1)
      e_leg = jnp.mean((qj[:, LEGS] - q_ref[:, LEGS]) ** 2, -1)
      T["ref_upper"] = rc.ref_upper * jnp.exp(-e_up / 0.5) * imitate
      T["ref_legs"] = rc.ref_legs * jnp.exp(-e_leg / 0.3) * imitate
      e_wr = jnp.mean((qj[:, WRIST] - q_ref[:, WRIST]) ** 2, -1)
      T["ref_wrist"] = rc.ref_wrist * jnp.exp(-e_wr / 0.5) * imitate * (g("face") < 0)
      tot["ref_rmse"] = tot["ref_rmse"] + jnp.where(imitate, jnp.sqrt(e_up), 0.0).sum()
      tot["ref_n"] = tot["ref_n"] + imitate.sum()
      cos_face = jnp.sum(bn * g("n_star"), -1)
      T["face"] = rc.face * jnp.exp(-(1 - cos_face) / 0.05) * (active & (tau > -0.02) & (tau < 0.12))
      T["face_coarse"] = rc.face_coarse * (1 - jnp.arccos(jnp.clip(cos_face, -1, 1)) / jnp.pi) * (
        active & (tau > -0.02) & (tau < 0.35))
      T["blade_vel"] = rc.blade_vel * jnp.exp(-(jnp.linalg.norm(bv - g("v_star"), axis=-1) / 0.4) ** 2) * (
        active & (tau > -0.03) & (tau < 0.08))
      imp = ev["imp"][s]
      T["impact"] = imp * (rc.impact_pos * jnp.exp(-(ev["imp_pos"][s] / 0.05) ** 2)
                           + rc.impact_pos_coarse * jnp.exp(-(ev["imp_pos"][s] / 0.15) ** 2)
                           + rc.impact_vel * jnp.exp(-(ev["imp_vel"][s] / 1.0) ** 2)
                           + rc.impact_ori * jnp.exp(-(1 - ev["imp_cos"][s]) / 0.1))
      tot["imp_pos_err"] = tot["imp_pos_err"] + jnp.where(imp, ev["imp_pos"][s], 0.0).sum()
      tot["imp_n"] = tot["imp_n"] + imp.sum()
      hit = ev["hit"][s] & ev["hit_face_ok"][s]
      shot_err = jnp.linalg.norm(ev["vout"][s] - g("vout_star"), axis=-1)
      T["hit"] = rc.hit * hit
      T["shot"] = rc.shot * jnp.exp(-(shot_err / 1.0) ** 2) * hit
      T["shot_outcome"] = rc.shot_outcome * jnp.exp(-(ev["shortfall"][s] / 0.25) ** 2) * hit
      T["playable"] = rc.playable * (ev["playable"][s] & hit)
      bh_hit = hit & (g("face") < 0)
      tot["bh_hits"] = tot["bh_hits"] + bh_hit.sum()
      tot["bh_playable"] = tot["bh_playable"] + (bh_hit & ev["playable"][s]).sum()
      tot["shot_err"] = tot["shot_err"] + jnp.where(hit, shot_err, 0.0).sum()
      tot["shortfall"] = tot["shortfall"] + jnp.where(hit, ev["shortfall"][s], 0.0).sum()
      land = ev["land"][s] & ev["land_face_ok"][s]
      T["land"] = rc.land * land + rc.land_acc * jnp.exp(-(ev["land_err"][s] / 0.3) ** 2) * land
      bh_scale = jnp.where(g("face") < 0, rc.bh_shot_scale, 1.0)
      for k in ("shot", "shot_outcome", "land"):
        T[k] = T[k] * bh_scale
      ready = jnp.exp(-(jnp.linalg.norm(blade - (base + C["ready_blade"]), axis=-1) / 0.2) ** 2)
      T["ready"] = rc.ready * ready * ~active * ~r_act
      T["upright"] = rc.upright * jnp.exp(-jnp.sum(grav[:, :2] ** 2, -1) / 0.05)
      T["alive"] = jnp.full(n, rc.alive, F32)
      stance = g("p_star")[:, :2] - C["ready_blade"][:2]
      off = stance - jnp.array([ROBOT_X, 0.0], F32)
      off_n = jnp.linalg.norm(off, axis=-1, keepdims=True)
      stance = jnp.array([ROBOT_X, 0.0], F32) + off * jnp.minimum(1.0, 0.8 / jnp.maximum(off_n, 1e-6))
      moving = active & (tau > 0.0) & (tau < 0.9)
      T["position"] = rc.position * jnp.exp(-(jnp.linalg.norm(base[:, :2] - stance, axis=-1) / 0.25) ** 2) * moving
      drift = jnp.hypot(base[:, 0] - ROBOT_X, base[:, 1])
      idle = ~(my_turn & g("plan_valid") & ~g("hit_done"))
      T["drift"] = rc.drift * jnp.maximum(drift - e.move_free, 0.0) ** 2 * idle
      T["action_rate"] = rc.action_rate * jnp.sum((g("last_action") - g("prev_action")) ** 2, -1)
      T["torque"] = rc.torque * jnp.sum(torque ** 2, -1)
      T["leg_vel"] = rc.leg_vel * jnp.sum(qdj[:, :12] ** 2, -1)
      span = C["jnt_hi"] - C["jnt_lo"]
      lo, hi = C["jnt_lo"] + 0.05 * span, C["jnt_hi"] - 0.05 * span
      T["joint_limit"] = rc.joint_limit * jnp.sum(jnp.maximum(lo - qj, 0) + jnp.maximum(qj - hi, 0), -1)
      over = (jnp.abs(blade[:, 0]) < TABLE_LEN / 2) & (jnp.abs(blade[:, 1]) < TABLE_WID / 2)
      T["paddle_table"] = rc.paddle_table * (over & (blade[:, 2] < TABLE_H + 0.03))
      T["fall"] = rc.fall * fell2[s]
      rew.append(sum(T.values()))
      for k, v in T.items():
        terms[k] = terms.get(k, 0.0) + v.astype(F32).mean() / 2
      tot["hits"] = tot["hits"] + hit.sum()
      tot["lands"] = tot["lands"] + land.sum()
      tot["land_err"] = tot["land_err"] + jnp.where(land, ev["land_err"][s], 0.0).sum()
      tot["faults"] = tot["faults"] + ev["fault"][s].sum()
      tot["misses"] = tot["misses"] + ev["miss"][s].sum()
    S = dict(S)
    S["side"] = side
    return S, jnp.concatenate(rew).astype(F32), terms, tot

  # ---------------------------------------------------------------- observations
  def _observe(self, C, S):
    """Duel.observe without noise: (2n, ACTOR_DIM), robot A rows then B."""
    ref = C["ref"]
    side = S["side"]
    out = []
    t = S["t"]
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(S, s)
      g = lambda k: side[k][s]
      tau = g("t_hit") - t
      act = g("plan_valid") & ~g("hit_done") & (tau > -0.06)
      p = jnp.where(act[:, None], g("p_star"), base + C["ready_blade"])
      nrm = jnp.where(act[:, None], g("n_star"), C["ready_normal"])
      vel = jnp.where(act[:, None], g("v_star"), 0.0)
      tau_c = jnp.where(act, jnp.clip(tau, -0.1, 1.0), 1.0)
      command = jnp.concatenate([_rot_yaw(yaw, p - base), _rot_yaw(yaw, nrm), 0.3 * _rot_yaw(yaw, vel),
                                 tau_c[:, None], act[:, None].astype(F32)], -1)
      side, r_act, rel = self._window(side, s, t, REF_LOOKAHEAD)
      q = ref.joints(g("ref_clip"), jnp.where(r_act, rel, 0.0), g("ref_stretch"))
      q = jnp.where(r_act[:, None], q, C["q_default"])
      ref_obs = jnp.concatenate([(q - C["q_default"])[:, UPPER], r_act[:, None].astype(F32)], -1)
      off = jnp.stack([base[:, 0] - ROBOT_X, base[:, 1], jnp.zeros_like(base[:, 0])], -1)
      home = _rot_yaw(yaw, off)[:, :2]
      side, p_act, rel = self._window(side, s, t, REF_LOOKAHEAD)
      r = jnp.where(p_act, rel, 0.0)
      pt = ref.blade_path(g("ref_clip"), r, g("ref_stretch"), g("ref_shift"))
      v_h = ref.imp_vel[g("ref_clip")] / g("ref_stretch")[:, None]
      pt = pt + (g("v_star") - v_h) * (r * jnp.exp(-(r / 0.15) ** 2))[:, None] + jnp.array([ROBOT_X, 0.0, 0.0], F32)
      path = _rot_yaw(yaw, jnp.where(p_act[:, None], pt - base, 0.0))
      out.append(jnp.concatenate([0.25 * gyro, grav, qj - C["q_default"], 0.05 * qdj, g("last_action"),
                                  command, ref_obs, home, path], -1).astype(F32))
    S = dict(S)
    S["side"] = side
    return S, jnp.concatenate(out)

  def _critic_obs(self, C, S, obs):
    n = self.cfg.n
    prm = C["prm"]
    out = []
    phys = jnp.stack([prm[0] / 0.126, prm[1] / 2.8e-3, prm[2], prm[3], prm[4], prm[5]], -1)
    for s in range(2):
      base, yaw, gyro, grav, qj, qdj = self._state(S, s)
      lin = flip(S["d"].qvel[:, 35 * s:35 * s + 3], s)
      alive = S["live"][:, None]
      ball_rel = jnp.where(alive, _rot_yaw(yaw, flip(S["bp"], s) - base), 0.0)
      mine = S["live"] & (S["to"] == s)
      phase = jnp.where(mine & S["bounced"], 2, jnp.where(mine, 1, jnp.where(S["live"], 3, 0)))
      priv = jnp.concatenate([
        _rot_yaw(yaw, lin), base[:, 2:3], ball_rel, 0.2 * _rot_yaw(yaw, flip(S["bv"], s)) * alive,
        0.005 * flip(S["bw"], s) * alive, _rot_yaw(yaw, flip(S["blade"][s], s) - base),
        0.3 * _rot_yaw(yaw, flip(S["blade_v"][s], s)), phys, jax.nn.one_hot(phase, 4, dtype=F32)], -1)
      a = obs[s * n:(s + 1) * n]
      cut = 3 + 3 + 29 * 3 + 11
      out.append(jnp.concatenate([a[:, :cut], priv, a[:, cut:]], -1))
    c = jnp.concatenate(out)
    assert c.shape[1] == CRITIC_DIM
    return c

  # ---------------------------------------------------------------- resets
  def _finish(self, C, S, fell2, timeout, key, tot):
    n = self.cfg.n
    fell = fell2.any(0)
    timer_new = jax.random.uniform(key, (n,), F32, 0.6, 1.2)
    for reason, mask in (("fall", fell), ("time", timeout & ~fell)):
      S, tot = self._end_rally(S, mask & (S["live"] | (S["rally_len"] > 0)), reason, timer_new, tot)
    reset = fell | timeout
    S = dict(S)
    S["d"] = self._reset_data(S["d"], reset)
    blade0, bn0 = self._read_blades(S["d"])
    r3 = reset[None, :, None]
    S["blade"] = jnp.where(r3, blade0, S["blade"])
    S["blade_n"] = jnp.where(r3, bn0, S["blade_n"])
    S["blade_v"] = jnp.where(r3, 0.0, S["blade_v"])
    S["steps"] = jnp.where(reset, 0, S["steps"])
    # Times restart with the match (all of them are relative): clear what used them.
    side = self._clear(S["side"], reset)
    side = dict(side)
    side["last_action"] = jnp.where(reset[None, :, None], 0.0, side["last_action"])
    S["side"] = side
    S["t"] = jnp.where(reset, 0.0, S["t"])
    S["need_plan"] = S["need_plan"] & ~reset
    S["replan_due"] = S["replan_due"] & ~reset
    return S, tot


TOT_KEYS = ("balls", "hits", "lands", "faults", "misses", "falls", "episodes", "imp_pos_err", "imp_n", "land_err",
            "ref_rmse", "ref_n", "rallies", "rally_sum", "rally_max", "shot_err", "shortfall", "bh_hits",
            "bh_playable", "serves", "cmd_bh", "cmd_bh_on_bh", "unreachable")
