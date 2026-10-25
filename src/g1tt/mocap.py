"""TTMD6 human table tennis strokes, cleaned and retargeted onto the G1.

    .venv/bin/python -m g1tt.mocap            # writes data/ttmd6_g1.npz

TTMD6 (Shanghai University of Sport, Sci. Rep. 14:3549, 2024, CC BY 4.0)
has 12,000 strokes of 6 types by 40 right-handed players: 14 joints and the
paddle centre at 120 Hz, millimetres, z up. Joint order, inferred from the
data: 0 hip centre, 1 neck/head, 2-4 left shoulder/elbow/wrist, 5-7 right
shoulder/elbow/wrist (paddle hand), 8-10 left hip/knee/ankle, 11-13 right
hip/knee/ankle. Lost markers are stored as (0, 0, 0).

Cleaning: zeros and implausible jumps become gaps, gaps up to 0.1 s are
interpolated, and a window around the impact (the paddle speed peak) must
be complete. Retargeting keeps each human bone's direction and gives it the
G1's length, chained from the hip centre out to the feet and to the paddle,
then solves whole-body IK for those points frame by frame. The paddle point
is what fixes the wrist orientation, which a 14-joint skeleton lacks.
"""

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from g1tt.model import ROOT, build_model, ready_qpos

DATA = ROOT / "data" / "ttmd6" / "TTMD6"
OUT = ROOT / "data" / "ttmd6_g1.npz"

SRC_HZ = 120
DST_HZ = 50
PRE_S = 0.6  # window before impact
POST_S = 0.3  # and after
MAX_GAP = 12  # frames at 120 Hz
CLASSES = {1: "fh_drive", 2: "fh_attack", 3: "fh_push", 4: "bh_drive", 5: "bh_attack", 6: "bh_push"}

# Human joint index -> G1 body whose origin plays that joint.
HIP_C, NECK, L_SH, L_EL, L_WR, R_SH, R_EL, R_WR, L_HIP, L_KN, L_AN, R_HIP, R_KN, R_AN = range(14)
G1_POINTS = {
  L_HIP: "left_hip_roll_link", L_KN: "left_knee_link", L_AN: "left_ankle_roll_link",
  R_HIP: "right_hip_roll_link", R_KN: "right_knee_link", R_AN: "right_ankle_roll_link",
  L_SH: "left_shoulder_roll_link", L_EL: "left_elbow_link", L_WR: "left_wrist_yaw_link",
  R_SH: "right_shoulder_roll_link", R_EL: "right_elbow_link", R_WR: "right_wrist_yaw_link",
}
BAT = 14  # the paddle as a 15th point
# (parent, child) chains from the hip centre outwards.
CHAINS = [(HIP_C, L_HIP), (L_HIP, L_KN), (L_KN, L_AN), (HIP_C, R_HIP), (R_HIP, R_KN), (R_KN, R_AN),
          (HIP_C, L_SH), (L_SH, L_EL), (L_EL, L_WR), (HIP_C, R_SH), (R_SH, R_EL), (R_EL, R_WR), (R_WR, BAT)]
WEIGHTS = {L_HIP: 0.5, R_HIP: 0.5, L_KN: 0.5, R_KN: 0.5, L_AN: 1.0, R_AN: 1.0, L_SH: 0.5, R_SH: 0.5,
           L_EL: 0.6, R_EL: 0.8, L_WR: 0.4, R_WR: 1.0, BAT: 1.5}


# ---------------------------------------------------------------- loading
def _load(path):
  return np.loadtxt(path, delimiter=",", encoding="utf-8-sig", ndmin=2)


def parse_name(path):
  parts = os.path.basename(path)[:-4].split("_")
  return {"id": int(parts[2]), "cls": int(parts[3]), "player": int(parts[4])}


def _fill(x, bad, max_gap):
  """Linear interpolation over bad frames of x (T, 3). None if a gap is too long."""
  if not bad.any():
    return x
  if bad.all():
    return None
  t = np.arange(len(x))
  runs = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
  starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
  if (ends - starts).max() > max_gap:
    return None
  good = ~bad
  return np.stack([np.interp(t, t[good], x[good, k]) for k in range(3)], -1)


def _smooth(x, width=5):
  k = np.exp(-0.5 * (np.arange(width) - width // 2) ** 2 / (width / 4) ** 2)
  k /= k.sum()
  pad = np.pad(x, ((width // 2, width // 2), (0, 0)), mode="edge")
  return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], -1)


def clean_stroke(hum_path):
  """Cleaned (T, 15, 3) metres (14 joints + paddle) cut around impact, or None."""
  H = _load(hum_path).reshape(-1, 14, 3) / 1000.0
  B = _load(hum_path.replace("TTMD_cut_hum", "TTMD_cut_bat").replace("human_", "bat_")) / 1000.0
  if len(B) != len(H):  # a few strokes pair body and paddle files of different lengths
    return None
  P = np.concatenate([H, B[:, None]], 1)
  T = len(P)
  out = np.empty_like(P)
  for j in range(15):
    x = P[:, j]
    bad = np.abs(x).sum(-1) == 0
    step = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    jump = step > (0.30 if j == BAT else 0.15)  # 36 / 18 m/s between frames
    bad[1:] |= jump
    bad[:-1] |= jump
    f = _fill(x, bad, MAX_GAP)
    if f is None:
      return None
    out[:, j] = f
  out = _smooth(out.reshape(T, -1)).reshape(T, 15, 3)
  speed = np.linalg.norm(np.diff(out[:, BAT], axis=0), axis=-1) * SRC_HZ
  if speed.max() > 25.0:
    return None
  imp = int(speed.argmax()) + 1
  a, b = imp - int(PRE_S * SRC_HZ), imp + int(POST_S * SRC_HZ)
  if a < 0 or b > T:
    return None
  return out[a:b], int(PRE_S * SRC_HZ), float(speed.max())


def resample(x, src_hz, dst_hz):
  T = len(x)
  t_src = np.arange(T) / src_hz
  t_dst = np.arange(0, t_src[-1] + 1e-9, 1 / dst_hz)
  flat = x.reshape(T, -1)
  return np.stack([np.interp(t_dst, t_src, flat[:, k]) for k in range(flat.shape[1])], -1).reshape(
    len(t_dst), *x.shape[1:])


# ---------------------------------------------------------------- retarget
class Retargeter:
  def __init__(self):
    self.m = build_model()
    self.d = mujoco.MjData(self.m)
    self.q0 = ready_qpos(self.m)
    self.bid = {j: self.m.body(b).id for j, b in G1_POINTS.items()}
    self.sid_bat = self.m.site("blade").id
    d = self.d
    d.qpos[:] = self.q0
    mujoco.mj_forward(self.m, d)
    pts = self._points()
    self.g1_len = {(a, b): np.linalg.norm(pts[b] - pts[a]) for a, b in CHAINS}
    jid = [self.m.jnt_qposadr[i] for i in range(1, self.m.njnt)]
    self.lo = self.m.jnt_range[1:, 0]
    self.hi = self.m.jnt_range[1:, 1]
    self.qadr = np.array(jid)

  def _points(self):
    pts = {j: self.d.xpos[b].copy() for j, b in self.bid.items()}
    pts[HIP_C] = 0.5 * (pts[L_HIP] + pts[R_HIP])
    pts[BAT] = self.d.site_xpos[self.sid_bat].copy()
    return pts

  def targets(self, P, origin):
    """G1 target points for one frame of human points P (15, 3) in the robot's frame."""
    tgt = {HIP_C: origin}
    for a, b in CHAINS:
      v = P[b] - P[a]
      tgt[b] = tgt[a] + v / max(np.linalg.norm(v), 1e-6) * self.g1_len[(a, b)]
    return tgt

  def solve(self, tgt, iters):
    m, d = self.m, self.d
    keys = [k for k in WEIGHTS]
    jac = np.zeros((3, m.nv))
    for _ in range(iters):
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      pts = self._points()
      J, e = [], []
      for k in keys:
        w = WEIGHTS[k]
        if k == BAT:
          mujoco.mj_jacSite(m, d, jac, None, self.sid_bat)
        else:
          mujoco.mj_jacBody(m, d, jac, None, self.bid[k])
        J.append(w * jac.copy())
        e.append(w * (tgt[k] - pts[k]))
      # Hip centre: average of the two hip Jacobians.
      jl, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
      mujoco.mj_jacBody(m, d, jl, None, self.bid[L_HIP])
      mujoco.mj_jacBody(m, d, jr, None, self.bid[R_HIP])
      J.append(2.0 * 0.5 * (jl + jr))
      e.append(2.0 * (tgt[HIP_C] - pts[HIP_C]))
      J, e = np.concatenate(J), np.concatenate(e)
      # Damped least squares with a weak pull to the ready pose.
      lam = 1e-2
      reg = np.zeros(m.nv)
      reg[6:] = 0.02 * (self.q0[7:] - d.qpos[7:])
      dq = np.linalg.solve(J.T @ J + lam * np.eye(m.nv), J.T @ e + lam * reg)
      mujoco.mj_integratePos(m, d.qpos, dq, 1.0)
      d.qpos[7:] = np.clip(d.qpos[7:], self.lo, self.hi)
    mujoco.mj_kinematics(m, d)
    pts = self._points()
    return float(np.mean([np.linalg.norm(tgt[k] - pts[k]) for k in keys]))

  def clip(self, P):
    """Retarget a (T, 15, 3) clip already in the robot's frame. Returns qpos (T, nq), blade (T, 3), err."""
    d = self.d
    d.qpos[:] = self.q0
    T = len(P)
    Q = np.zeros((T, self.m.nq))
    blade = np.zeros((T, 3))
    errs = []
    for t in range(T):
      # Hip centre: the human's, scaled by leg length, over the G1's start position.
      tgt = self.targets(P[t], self.origin[t])
      if t == 0:
        d.qpos[0:3] = self.origin[0] + (self.q0[0:3] - self.g1_hipc0)
      errs.append(self.solve(tgt, 60 if t == 0 else 12))
      Q[t] = d.qpos
      blade[t] = d.site_xpos[self.sid_bat]
    return Q, blade, float(np.mean(errs))


def to_robot_frame(P):
  """Rotate and translate a clip so the player faces +x with the hip centre at
  the G1's hip centre, and scale vertical position by leg length."""
  start = P[:8].mean(0)
  across = start[R_HIP] - start[L_HIP]
  across[2] = 0
  across /= np.linalg.norm(across)
  fwd = np.cross([0.0, 0.0, 1.0], across)  # facing direction, right hip on -y side
  left = -across
  R = np.stack([fwd, left, [0, 0, 1.0]])  # rows: new x, y, z
  return (P - start[HIP_C]) @ R.T


def _work(args):
  path, = args
  meta = parse_name(path)
  c = clean_stroke(path)
  if c is None:
    return None
  P, imp, peak = c
  P = to_robot_frame(P)
  P50 = resample(P, SRC_HZ, DST_HZ)
  imp50 = int(round(imp * DST_HZ / SRC_HZ))
  rt = _get_rt()
  # Hip centre trajectory: human displacement scaled by leg length ratio.
  leg_h = np.median(np.linalg.norm(P50[:, R_HIP] - P50[:, R_AN], axis=-1))
  leg_g = rt.g1_len[(R_HIP, R_KN)] + rt.g1_len[(R_KN, R_AN)]
  rt.origin = rt.g1_hipc0 + (P50[:, HIP_C] - P50[0, HIP_C]) * (leg_g / leg_h)
  Q, blade, err = rt.clip(P50)
  return dict(meta, q=Q.astype(np.float32), blade=blade.astype(np.float32), impact=imp50, peak_speed=peak,
              err=err, human=P50.astype(np.float32))


_RT = None


def _get_rt():
  global _RT
  if _RT is None:
    _RT = Retargeter()
    d = _RT.d
    d.qpos[:] = _RT.q0
    mujoco.mj_kinematics(_RT.m, d)
    _RT.g1_hipc0 = _RT._points()[HIP_C]
  return _RT


def build(limit_per_class: int, jobs: int, classes):
  files = sorted(glob.glob(str(DATA / "TTMD_cut_hum" / "*.csv")))
  by_cls = {}
  for f in files:
    by_cls.setdefault(parse_name(f)["cls"], []).append(f)
  chosen = []
  rng = np.random.default_rng(0)
  for k in classes:
    fs = by_cls[k]
    chosen += [fs[i] for i in rng.permutation(len(fs))[: limit_per_class * 3]]
  with Pool(jobs) as p:
    res = [r for r in p.imap(_work, [(f,) for f in chosen], chunksize=8) if r is not None]
  out = []
  for k in classes:
    out += [r for r in res if r["cls"] == k][:limit_per_class]
  return out


def save(clips, path=OUT):
  lens = np.array([len(c["q"]) for c in clips])
  np.savez_compressed(
    path,
    q=np.concatenate([c["q"] for c in clips]),
    blade=np.concatenate([c["blade"] for c in clips]),
    human=np.concatenate([c["human"] for c in clips]),
    offsets=np.concatenate([[0], np.cumsum(lens)[:-1]]),
    lengths=lens,
    impact=np.array([c["impact"] for c in clips]),
    cls=np.array([c["cls"] for c in clips]),
    player=np.array([c["player"] for c in clips]),
    clip_id=np.array([c["id"] for c in clips]),
    peak_speed=np.array([c["peak_speed"] for c in clips]),
    err=np.array([c["err"] for c in clips]),
    hz=DST_HZ,
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--per-class", type=int, default=300)
  ap.add_argument("--classes", default="1,3,4,6", help="1 fh drive, 2 fh attack, 3 fh push, 4-6 backhand")
  ap.add_argument("--jobs", type=int, default=8)
  args = ap.parse_args()
  classes = [int(c) for c in args.classes.split(",")]
  clips = build(args.per_class, args.jobs, classes)
  save(clips)
  for k in classes:
    cs = [c for c in clips if c["cls"] == k]
    if cs:
      print(f"class {k} {CLASSES[k]:10s} clips {len(cs):4d} retarget err {np.mean([c['err'] for c in cs]):.3f} m "
            f"human peak paddle {np.median([c['peak_speed'] for c in cs]):.1f} m/s")
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
