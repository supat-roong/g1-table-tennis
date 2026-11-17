"""Train on a Kaggle GPU: publish the training notebook and start it.

    python tools/kaggle/push.py --user <kaggle-user> --name g1
    python tools/kaggle/push.py --user <kaggle-user> --name g1 --resume   # next session

The notebook (train_notebook.py, converted to .ipynb) is public and clones the
code from GitHub at the current commit, so push your commits first. It attaches
the public strokes dataset (--strokes, default supatroongpraiwan/ttmd6-g1-strokes).
With --resume, the new version attaches the notebook's own latest output
(runs/<name>/ with ckpt.pt and log.jsonl) and continues from it, so a run is
chained one session at a time with nothing uploaded. --resume-from DIR
continues a local run folder instead, uploaded as the private dataset
<user>/g1tt-runs. Download the result with
`kaggle kernels output <user>/unitree-g1-humanoids-learn-table-tennis-rallies -p runs/kaggle`.
Needs the kaggle CLI (pip install kaggle, or set KAGGLE_CLI="uvx kaggle") and a
token in ~/.kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = Path(__file__).with_name("train_notebook.py")
TITLE = "Unitree G1 Humanoids Learn Table Tennis Rallies"
SLUG = "unitree-g1-humanoids-learn-table-tennis-rallies"  # Kaggle derives the URL from the title
KEYWORDS = ["reinforcement learning", "robotics", "pytorch", "jax", "simulations"]  # Kaggle allows at most 5 tags
STROKES = "supatroongpraiwan/ttmd6-g1-strokes"  # TTMD6 strokes retargeted onto the G1 (python -m g1tt.mocap)


def kaggle(*args: str) -> str:
  cmd = shlex.split(os.environ.get("KAGGLE_CLI", "kaggle")) + list(args)
  r = subprocess.run(cmd, capture_output=True, text=True)
  out = (r.stdout + r.stderr).strip()
  print(out.splitlines()[-1] if out else "", flush=True)
  return out


def git(*args: str) -> str:
  return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dataset_ready(ref: str) -> bool:
  return "ready" in kaggle("datasets", "status", ref).lower()


def upload_dataset(folder: Path, ref: str, title: str, message: str, license_name: str):
  (folder / "dataset-metadata.json").write_text(json.dumps({"title": title, "id": ref,
                                                            "licenses": [{"name": license_name}]}))
  if dataset_ready(ref):
    kaggle("datasets", "version", "-p", str(folder), "-m", message, "-q")
  else:
    kaggle("datasets", "create", "-p", str(folder), "-q")
  time.sleep(10)
  for _ in range(60):  # a new version must finish processing before a notebook can mount it
    if dataset_ready(ref):
      break
    time.sleep(10)
  time.sleep(20)


def to_ipynb(source: str) -> dict:
  """Convert a "# %%" percent-format script to a notebook; "# %% [markdown]" cells lose their "# "."""
  cells = []
  for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    markdown = "[markdown]" in header
    lines = body.strip("\n").splitlines()
    if markdown:
      lines = [re.sub(r"^# ?", "", ln) for ln in lines]
    text = [ln + "\n" for ln in lines]
    if text:
      text[-1] = text[-1].rstrip("\n")
    cell = {"cell_type": "markdown" if markdown else "code", "metadata": {}, "source": text}
    if not markdown:
      cell.update(execution_count=None, outputs=[])
    cells.append(cell)
  return {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}}}


def set_setting(source: str, name: str, value) -> str:
  text = json.dumps(value) if isinstance(value, str) else repr(value)
  return re.sub(rf"^{name} = .*?(  #.*)?$", lambda m: f"{name} = {text}{m.group(1) or ''}", source, count=1, flags=re.M)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--user", required=True, help="Kaggle username")
  p.add_argument("--name", default="g1", help="run name (runs/<name>/)")
  p.add_argument("--args", default="", help="extra g1tt.gpu.train_gpu arguments")
  p.add_argument("--max-hours", type=float, default=11.0, help="stop and save ckpt.pt before Kaggle's 12 h limit")
  p.add_argument("--resume", action="store_true", help="continue from this notebook's latest output")
  p.add_argument("--resume-from", help="local run folder to continue (with ckpt.pt), uploaded as a private dataset")
  p.add_argument("--private", action="store_true", help="keep the notebook private (default: public)")
  p.add_argument("--strokes", default=STROKES, help="Kaggle dataset with ttmd6_g1.npz")
  a = p.parse_args()

  rev = git("rev-parse", "HEAD")
  if not git("branch", "-r", "--contains", rev):
    sys.exit(f"commit {rev[:7]} is not on any remote branch; the notebook clones from GitHub, so git push first")
  repo = re.sub(r"^git@github\.com:", "https://github.com/", git("remote", "get-url", "origin")).removesuffix(".git")

  with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    sources, kernels = [a.strokes], []

    if a.resume:
      kernels.append(f"{a.user}/{SLUG}")
    elif a.resume_from:
      run_dir = Path(a.resume_from)
      assert (run_dir / "ckpt.pt").exists(), f"no {run_dir}/ckpt.pt"
      dst = tmp / "runs" / "runs" / a.name
      dst.mkdir(parents=True)
      for f in ("ckpt.pt", "log.jsonl", "cfg.json"):  # resume state and logs; not every snapshot
        if (run_dir / f).exists():
          shutil.copy(run_dir / f, dst)
      sources.append(f"{a.user}/g1tt-runs")
      upload_dataset(tmp / "runs", sources[1], "g1tt-runs", f"resume {a.name}", "other")

    code = NOTEBOOK.read_text()
    for name, value in [("REPO", repo), ("REF", rev), ("RUN_NAME", a.name), ("TRAIN_ARGS", a.args),
                        ("MAX_HOURS", a.max_hours), ("RESUME", bool(a.resume or a.resume_from))]:
      code = set_setting(code, name, value)
    kdir = tmp / "kernel"
    kdir.mkdir()
    (kdir / f"{SLUG}.ipynb").write_text(json.dumps(to_ipynb(code), indent=1))
    (kdir / "kernel-metadata.json").write_text(json.dumps({
      "id": f"{a.user}/{SLUG}", "title": TITLE, "code_file": f"{SLUG}.ipynb",
      "language": "python", "kernel_type": "notebook", "is_private": a.private, "enable_gpu": True,
      "enable_internet": True, "dataset_sources": sources, "kernel_sources": kernels, "keywords": KEYWORDS,
      "machine_shape": "NvidiaTeslaT4",
    }))
    kaggle("kernels", "push", "-p", str(kdir))
  print(f"progress: https://www.kaggle.com/code/{a.user}/{SLUG}")


if __name__ == "__main__":
  main()
