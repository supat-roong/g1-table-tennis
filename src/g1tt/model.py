"""Build the MuJoCo scene: Unitree G1 holding a paddle, a regulation table, net and ball.

World frame: the table centre is the origin, +x points from the robot's end
towards the opponent's end, +z is up. The robot stands behind the -x end line
facing +x, so its right hand side is -y.

Only the robot is simulated by MuJoCo. The ball is a mocap body whose state
comes from `g1tt.ball`, and the paddle and table have no MuJoCo collisions
with it: a 2.7 g ball at 20+ m/s against a 2 mm robot timestep tunnels and
conditions the solver badly, so its contacts are resolved analytically.
"""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
G1_XML = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1_mjx.xml"

# ITTF dimensions (m).
TABLE_LEN = 2.74
TABLE_WID = 1.525
TABLE_H = 0.76
NET_H = 0.1525
NET_OVERHANG = 0.1525

# Paddle (shakehand grip). Blade centre in the right_wrist_yaw_link frame;
# the blade lies in that link's x-z plane, so its face normal is the link's y.
BLADE_RADIUS = 0.076
BLADE_THICK = 0.01
BLADE_POS = np.array([0.22, 0.0, 0.0])

ROBOT_X = -TABLE_LEN / 2 - 0.47  # hit targets sit at ~85% of arm reach
ROBOT_STANCE = "knees_bent"

JOINTS = [
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
  "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
  "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
  "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
  "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
  "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# Ready stance: knees bent, paddle held in front of the right hip.
READY_POSE = {
  "left_hip_pitch": -0.312, "left_knee": 0.669, "left_ankle_pitch": -0.363,
  "right_hip_pitch": -0.312, "right_knee": 0.669, "right_ankle_pitch": -0.363,
  "waist_pitch": 0.073,
  "left_shoulder_pitch": 0.2, "left_shoulder_roll": 0.22, "left_elbow": 1.0,
  # Right arm from a small IK solve: blade ~0.13 m above the table, in front
  # of the right hip, forehand face towards the opponent. Elbow 0 is a
  # horizontal forearm on the G1; positive extends it downwards.
  "right_shoulder_pitch": -0.2, "right_shoulder_roll": -0.14, "right_shoulder_yaw": -0.12,
  "right_elbow": -0.05, "right_wrist_roll": 0.38, "right_wrist_pitch": -0.1, "right_wrist_yaw": -0.91,
}


def _add_scene(spec: mujoco.MjSpec) -> None:
  world = spec.worldbody
  spec.option.timestep = 0.004

  spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.25, 0.27, 0.3], rgb2=[0.2, 0.22, 0.25], width=300, height=300)
  spec.add_material(name="floor", textures=["", "grid"], texuniform=True, texrepeat=[6, 6])
  spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                   rgb1=[0.35, 0.45, 0.6], rgb2=[0.05, 0.05, 0.08], width=512, height=3072)
  world.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)

  world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor",
                 contype=0, conaffinity=0)

  # Table: playing surface, white lines, legs. Visual only for MuJoCo.
  top = world.add_body(name="table", pos=[0, 0, TABLE_H])
  vis = dict(contype=0, conaffinity=0)
  top.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[TABLE_LEN / 2, TABLE_WID / 2, 0.015],
               pos=[0, 0, -0.015], rgba=[0.05, 0.2, 0.45, 1], **vis)
  for name, size, pos in [
    ("line_centre", [TABLE_LEN / 2, 0.0015, 0.0005], [0, 0, 0]),
    ("line_left", [TABLE_LEN / 2, 0.01, 0.0005], [0, TABLE_WID / 2 - 0.01, 0]),
    ("line_right", [TABLE_LEN / 2, 0.01, 0.0005], [0, -TABLE_WID / 2 + 0.01, 0]),
    ("line_near", [0.01, TABLE_WID / 2, 0.0005], [-TABLE_LEN / 2 + 0.01, 0, 0]),
    ("line_far", [0.01, TABLE_WID / 2, 0.0005], [TABLE_LEN / 2 - 0.01, 0, 0]),
  ]:
    top.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=[1, 1, 1, 1], **vis)
  for sx in (-1, 1):
    for sy in (-1, 1):
      top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, TABLE_H / 2],
                   pos=[sx * (TABLE_LEN / 2 - 0.2), sy * (TABLE_WID / 2 - 0.1), -TABLE_H / 2],
                   rgba=[0.15, 0.15, 0.15, 1], **vis)
  top.add_geom(name="net", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[0.002, TABLE_WID / 2 + NET_OVERHANG, NET_H / 2], pos=[0, 0, NET_H / 2],
               rgba=[0.9, 0.9, 0.9, 0.6], **vis)

  ball = world.add_body(name="ball", mocap=True, pos=[1.0, 0, 1.0])
  ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[1, 0.55, 0.1, 1], **vis)

  # Planner targets, drawn for debugging: where the paddle should be at impact.
  tgt = world.add_body(name="hit_target", mocap=True, pos=[0, 0, -1])
  tgt.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[0.1, 1, 0.3, 0.4], **vis)

  # Foot-floor contacts. The menagerie MJX model disables default contacts and
  # lists pairs explicitly; the floor here is a new geom, so pair it again.
  for side in ("left", "right"):
    for i in (1, 2, 3):
      spec.add_pair(geomname1=f"{side}_foot{i}_collision", geomname2="floor", condim=3,
                    solref=[0.008, 1], friction=[1, 1, 0.005, 0.0001, 0.0001])


def _add_paddle(spec: mujoco.MjSpec) -> None:
  wrist = spec.body("right_wrist_yaw_link")
  paddle = wrist.add_body(name="paddle", pos=[0.08, 0, 0])
  # Handle in the hand, blade beyond the fingers. ~170 g, a typical paddle.
  paddle.add_geom(name="paddle_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.013, 0.016], pos=[0.02, 0, 0],
                  mass=0.07, rgba=[0.55, 0.35, 0.15, 1], contype=0, conaffinity=0, group=2)
  bx = BLADE_POS[0] - 0.08
  blade_quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]  # cylinder axis (z) -> link y
  paddle.add_geom(name="blade", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BLADE_RADIUS, BLADE_THICK / 2, 0],
                  pos=[bx, 0, 0], quat=blade_quat, mass=0.10, rgba=[0.55, 0.35, 0.15, 1],
                  contype=0, conaffinity=0, group=2)
  # The two rubbers: red forehand face on +y, black backhand face on -y.
  for name, y, rgba in (("rubber_fh", 1, [0.8, 0.1, 0.1, 1]), ("rubber_bh", -1, [0.05, 0.05, 0.05, 1])):
    paddle.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BLADE_RADIUS - 0.001, 0.001, 0],
                    pos=[bx, y * (BLADE_THICK / 2 + 0.001), 0], quat=blade_quat, mass=0.0, rgba=rgba,
                    contype=0, conaffinity=0, group=2)
  # Site at the blade centre, axes aligned with the link: x along the paddle,
  # y = forehand face normal, z = across the blade.
  paddle.add_site(name="blade", pos=[bx, 0, 0], size=[0.01, 0, 0])


def build_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.modelname = "g1_table_tennis"
  _add_scene(spec)
  _add_paddle(spec)
  pelvis = spec.body("pelvis")
  pelvis.pos = [ROBOT_X, 0, pelvis.pos[2]]
  return spec


def build_model() -> mujoco.MjModel:
  return build_spec().compile()


def ready_qpos(model: mujoco.MjModel) -> np.ndarray:
  """qpos for the robot standing in the ready stance behind its end line."""
  q = np.zeros(model.nq)
  q[0:3] = [ROBOT_X, 0, 0.755]
  q[3] = 1.0
  for name, val in READY_POSE.items():
    q[model.joint(f"{name}_joint").qposadr[0]] = val
  return q


def default_joint_targets() -> np.ndarray:
  return np.array([READY_POSE.get(j, 0.0) for j in JOINTS])


def write_xml(path: Path) -> None:
  spec = build_spec()
  spec.compile()
  path.write_text(spec.to_xml())


# ---------------------------------------------------------------- two robots
OPP_PREFIX = "b_"
YAW_180 = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion (w, x, y, z)


def build_duel_spec() -> mujoco.MjSpec:
  """Two G1s with paddles, one at each end, facing each other.

  Robot A is the usual one (behind the -x end); robot B is the same model
  turned 180 degrees about z behind the +x end, its names prefixed "b_". A
  rotation, not a mirror image: B is right-handed like A, and the world seen
  from B is A's world turned half a circle.
  """
  spec = build_spec()
  opp = mujoco.MjSpec.from_file(str(G1_XML))
  _add_paddle(opp)
  opp.body("pelvis").pos = [0, 0, opp.body("pelvis").pos[2]]
  frame = spec.worldbody.add_frame(pos=[-ROBOT_X, 0, 0], quat=YAW_180)
  spec.attach(opp, prefix=OPP_PREFIX, frame=frame)
  for i in (1, 2, 3):
    for side in ("left", "right"):
      spec.add_pair(geomname1=f"{OPP_PREFIX}{side}_foot{i}_collision", geomname2="floor", condim=3,
                    solref=[0.008, 1], friction=[1, 1, 0.005, 0.0001, 0.0001])
  tgt = spec.worldbody.add_body(name="hit_target_b", mocap=True, pos=[0, 0, -1])
  tgt.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[0.3, 0.6, 1, 0.4],
               contype=0, conaffinity=0)
  return spec


def build_duel_model() -> mujoco.MjModel:
  return build_duel_spec().compile()
