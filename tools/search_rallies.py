"""Search single-match seeds for long robot-vs-robot rallies, in parallel.

    .venv/bin/python tools/search_rallies.py runs/g1/ckpt.pt --seeds 0-35 --seconds 300

Prints every rally of at least --min returns with its footwork hits; one match
per seed, so `python -m g1tt.duel <ckpt> --seed S --seconds T --video ...`
with --matches 1 replays any of them exactly.
"""
import argparse
import json
from multiprocessing import Pool

import torch

from g1tt.duel import run


def one(args):
  ckpt, seed, seconds, wide, bh = args
  torch.set_num_threads(1)
  _, log = run(ckpt, 1, seconds, seed, wide_frac=wide, bh_frac=bh)
  return [(seed, *r[1:]) for r in log]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("ckpt")
  ap.add_argument("--seeds", default="0-31", help="a range (0-31) or a list (3,7,12)")
  ap.add_argument("--seconds", type=float, default=300.0)
  ap.add_argument("--min", type=int, default=100)
  ap.add_argument("--jobs", type=int, default=9)
  ap.add_argument("--wide-frac", type=float, default=0.10)
  ap.add_argument("--bh-frac", type=float, default=0.25)
  a = ap.parse_args()
  if "-" in a.seeds:
    lo, hi = map(int, a.seeds.split("-"))
    seeds = range(lo, hi + 1)
  else:
    seeds = [int(x) for x in a.seeds.split(",")]
  with Pool(a.jobs) as p:
    rallies = [r for rs in p.imap_unordered(one, [(a.ckpt, s, a.seconds, a.wide_frac, a.bh_frac) for s in seeds]) for r in rs]
  # Rallies with footwork and both strokes first, then the longest.
  # Backhands and moves count only mid-rally: not the serve receive, and with
  # the rally carrying on after them (Duel.SOLID_AFTER).
  full = lambda r: r[6] > 0 and r[8] > 0 and r[9] > 0
  long = sorted([r for r in rallies if r[2] >= a.min], key=lambda r: (-full(r), -r[2]))
  print(json.dumps({"rallies": len(rallies), "longest": max((r[2] for r in rallies), default=0), f"ge_{a.min}": len(long),
                    f"ge_{a.min}_with_moves": sum(r[5] > 0 for r in long),
                    f"ge_{a.min}_solid_fh_bh_moves": sum(full(r) for r in long)}))
  for seed, t, k, why, wd, mv, fh, bh, sbh, smv in long[:15]:
    print(f"seed {seed:3d}  returns {k:4d}  hits after moving {mv:3d} (solid {smv})  wide-ball hits {wd:3d}  "
          f"forehand {fh:3d}  backhand {bh:3d} (solid {sbh})  ended at {t:6.1f} s by {why}")


if __name__ == "__main__":
  main()
