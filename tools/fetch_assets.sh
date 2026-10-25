#!/usr/bin/env bash
# Fetch the Unitree G1 model from MuJoCo Menagerie (BSD-3-Clause) into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
git -C third_party/mujoco_menagerie sparse-checkout set unitree_g1
