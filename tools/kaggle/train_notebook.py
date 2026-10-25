# %% [markdown]
# # Unitree G1 Humanoids Learn Table Tennis Rallies
#
# Two simulated Unitree G1 humanoids learn to rally each other at table tennis, in MuJoCo, with one
# shared policy trained from scratch by self-play. This notebook trains that policy on a Kaggle GPU.
#
# Code and full write-up: https://github.com/supat-roong/g1-table-tennis
#
# ## How it works
#
# The robot never sees the ball. A model-based **planner** (as in HITTER) predicts each incoming ball,
# picks where to meet it and where to send it, and turns that into a **paddle command**: where the
# blade should be, which way its face points, how fast it moves and when. A **whole-body policy**
# (PPO, 29 joints) learns to deliver that command while staying on its feet.
#
# | Part | What it does |
# |---|---|
# | Ball physics | Drag, Magnus lift, table and rubber bounces with spin, at 1 kHz |
# | Planner | Predicts the bounce, inverts the rubber model for the blade pose and velocity that land the return on target |
# | Human strokes | 300 forehand and 151 backhand drives from the TTMD6 motion-capture dataset, retargeted onto the G1; the policy is rewarded for following one near the planned contact, shifted onto it |
# | Self-play | Both robots run the same policy; each one's returns are the other's incoming balls |
# | Success Guided Sampling | Serves and return targets the robots succeed on about half the time are picked more often |
#
# On the GPU the whole match (two robots, ball, rally rules, rewards, planner) is one compiled JAX step
# over 2,048 matches, with the physics in MuJoCo Warp, and PPO runs in PyTorch on the same GPU.
#
# ## Run it yourself
#
# 1. The human strokes come from TTMD6 (Shanghai University of Sport, CC BY 4.0). Build them on your
#    machine (see "Human motion data" in the README) and upload `data/ttmd6_g1.npz` as a Kaggle dataset.
# 2. Copy this notebook, then in the side panel: **Add Input** your strokes dataset, set
#    **Accelerator** to GPU T4 x2, and turn **Internet** on (phone-verified account needed).
# 3. Edit the settings below if you like, then **Save Version → Save & Run All**.
#
# Kaggle stops a session after 12 hours; training saves and stops at `MAX_HOURS`. A session does about
# 7,000 iterations, and the full 21,000 take about 33 GPU hours: attach the previous output as a
# dataset and set `RESUME = True` to continue. Results land in the notebook output under `runs/<name>/`.
#
# ## Settings
#
# - `REF`: the commit this run uses, so the result is reproducible.
# - `TRAIN_ARGS`: any `g1tt.gpu.train_gpu` option; the defaults are the recipe.
#   `python -m g1tt.gpu.train_gpu --help` lists them.
# - `RESUME`: continue `runs/<RUN_NAME>/ckpt.pt` from an attached dataset.

# %%
REPO = "https://github.com/supat-roong/g1-table-tennis"
REF = "main"  # branch, tag or commit to train
RUN_NAME = "g1"
TRAIN_ARGS = ""  # extra g1tt.gpu.train_gpu arguments, e.g. "--iters 7000 --seed 1"
MAX_HOURS = 11.0  # save ckpt.pt and stop before Kaggle's 12 hour limit
RESUME = False  # continue runs/<RUN_NAME>/ckpt.pt from an attached dataset

# %% [markdown]
# ## 1. Get the code
#
# Clone the repository at `REF` and show the hardware.

# %%
import glob
import os
import shutil
import subprocess
import sys

PY = sys.executable
CODE = "/kaggle/tmp/g1-table-tennis"  # outside /kaggle/working, so only runs/ is saved as output
RUNS = "/kaggle/working/runs"
QUIET = ("iterations limit reached", "warn_overflow", "OverflowType", "load on device")  # MuJoCo Warp solver notices


def run(cmd, cwd=CODE, quiet=False):
    """Run a shell command, stream its output, and stop the notebook if it fails."""
    print(f"$ {cmd}", flush=True)
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "XLA_PYTHON_CLIENT_MEM_FRACTION": ".45"}  # JAX and PyTorch share the GPU
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if not (quiet and any(q in line for q in QUIET)):
            print(line, end="", flush=True)
    if p.wait():
        raise subprocess.CalledProcessError(p.returncode, cmd)


shutil.rmtree(CODE, ignore_errors=True)
run(f"git clone -q {REPO} {CODE} && git -C {CODE} checkout -q {REF} && git -C {CODE} log -1 --oneline", cwd="/kaggle")
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc")

# %% [markdown]
# ## 2. Install
#
# MuJoCo, MJX with MuJoCo Warp for the physics on the GPU, and JAX with CUDA. Kaggle's PyTorch already
# has CUDA. Then the Unitree G1 model from MuJoCo Menagerie (BSD-3-Clause).

# %%
run(f"{PY} -m pip install -q 'mujoco==3.14.0' 'mujoco-mjx==3.14.0' warp-lang 'jax[cuda12]' pytest")
run(f"{PY} -m pip install -q --no-deps -e .")
run("bash tools/fetch_assets.sh")
run(f"{PY} -c \"import jax, torch, mujoco; print(jax.devices(), torch.cuda.get_device_name(0), mujoco.__version__)\"")

# %% [markdown]
# ## 3. Find your human strokes
#
# Any attached dataset with a `ttmd6_g1.npz` works: the retargeted TTMD6 strokes, joint angles and
# paddle path at 50 Hz with each stroke's contact frame. It is copied to `data/ttmd6_g1.npz`.

# %%
found = glob.glob("/kaggle/input/**/ttmd6_g1.npz", recursive=True)
if not found:
    raise FileNotFoundError(
        "No ttmd6_g1.npz under /kaggle/input. Build it with g1tt.mocap (see the README), upload it as a "
        "dataset, and attach it with Add Input."
    )
os.makedirs(f"{CODE}/data", exist_ok=True)
shutil.copy(found[0], f"{CODE}/data/ttmd6_g1.npz")
print("strokes:", found[0])

# %% [markdown]
# ## 4. Check the GPU environment
#
# The compiled duel must compute exactly what the CPU one does. This test copies CPU match states into
# the JAX environment and compares observations, critic inputs and rewards.

# %%
run(f"{PY} -m pytest -q tests/test_gpu_duel.py tests/test_ball_jit.py")

# %% [markdown]
# ## 5. Train
#
# PPO with an asymmetric actor-critic (all defaults in `g1tt/train.py` and `g1tt/ppo.py`):
#
# | Choice | Setting | Why |
# |---|---|---|
# | Observations | Proprioception, the paddle command, the human stroke 0.1 s ahead; never the ball | What a real robot with motion capture of itself would have |
# | Critic | Also the ball, its spin and the randomised physics constants | Better values from state the actor cannot see |
# | Rewards | Hit, land on target, playable return; follow the paddle path, face and velocity at contact; follow the human stroke; stay up | Dense shaping to the planned contact, sparse credit for the outcome |
# | Curriculum | Serves start slow and nearly spinless and get harder once 40% are hit | A first hit is hard to find with full-speed spin |
# | Return mix | 65% to the partner's forehand, 25% to its backhand, 10% wide of its reach | Both strokes and footwork, in every rally |
# | Sampling | Success Guided Sampling picks half the serves and returns near 50% success | Practice where learning happens |
#
# Each log line is one update of 98,304 decisions (2,048 matches x 2 robots x 24 steps):
#
# - `hit`, `land`: share of balls hit, and landed on the other half
# - `rally`: mean returns per rally, `bhp`: backhands the partner can play
# - `sgs`: mean success of the return targets, and the share near 50%

# %%
os.makedirs(RUNS, exist_ok=True)
resume = ""
if RESUME:
    prev = glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}/ckpt.pt", recursive=True)
    if not prev:
        raise FileNotFoundError(f"RESUME is on but no runs/{RUN_NAME}/ckpt.pt is attached")
    shutil.copytree(os.path.dirname(prev[0]), f"{RUNS}/{RUN_NAME}", dirs_exist_ok=True)
    resume = "--resume"

run(f"{PY} -m g1tt.gpu.train_gpu --tag {RUN_NAME} --runs {RUNS} {TRAIN_ARGS} --max-hours {MAX_HOURS} {resume}",
    quiet=True)

# %% [markdown]
# ## 6. Training curves

# %%
import json

import matplotlib.pyplot as plt

rows = [json.loads(line) for line in open(f"{RUNS}/{RUN_NAME}/log.jsonl")]
it = [r["iter"] for r in rows]
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
ax[0].plot(it, [r["hit_rate"] for r in rows], label="hit")
ax[0].plot(it, [r["land_rate"] for r in rows], label="landed")
ax[0].set(xlabel="iteration", ylabel="share of balls", ylim=(0, 1))
ax[0].legend()
ax[1].plot(it, [r["rally_mean"] for r in rows])
ax[1].set(xlabel="iteration", ylabel="returns per rally")
ax[2].plot(it, [r["std"] for r in rows])
ax[2].set(xlabel="iteration", ylabel="action noise (std)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. A match
#
# The deterministic policy plays 4 matches of 60 s on the CPU with the full rules, aiming like in
# training. `rally_mean_returns` counts returns that landed; `ends` says how rallies ended.

# %%
run(f"{PY} -m g1tt.duel {RUNS}/{RUN_NAME}/ckpt.pt --matches 4 --seconds 60")

# %% [markdown]
# ## What next
#
# To watch your run, download the output (`runs/<name>/`) next to a clone of the repo, then:
#
# ```bash
# python -m g1tt.duel runs/g1/ckpt.pt --find-rally 16 --seconds 300   # films the longest rally
# python tools/search_rallies.py runs/g1/ckpt.pt --seeds 0-35          # many seeds, in parallel
# ```
