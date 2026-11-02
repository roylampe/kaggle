"""
Build packings.csv: a verified packing of n unit squares in a square, for n = 1..1000.

Columns: id, n, square_id, x, y, angle   (angle in radians; x, y = square centre).
The container for each n is the axis-aligned square [0, side] x [0, side]
(side is listed in packings_summary.csv).

Sources of each packing (cheapest container wins):
  * "grid"      : ceil(sqrt(n)) x ceil(sqrt(n)) axis-aligned grid.
  * "optimizer" : small-n packings found by square_packing.py (seeds.json).
  * "L-extend"  : take a packing of n0 squares in side s0 and add an "L" of
                  axis-aligned squares along two sides -> side s0 + 1 holds
                  n0 + 2*floor(s0) + 1 squares.  Applied recursively.
  * "truncate"  : drop squares from a packing of more squares.
Squares with different angles are separated by a gap of 1e-6; squares with the same angle
(e.g. every grid) sit exactly flush (coordinates are exact multiples of 2**-40).
Every packing is checked with an exact separating-axis test (reading the written
CSV back from disk) before the script finishes.
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.setrecursionlimit(20000)
N_MAX = 1000
EPS = 1e-12
GAP = 1e-6          # gap between squares with DIFFERENT angles; same-angle squares are flush
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strict_gaps import Q, finalize, fmt, half_extent, text_safe, tidy_gaps, validate      # noqa: E402


# ---------------------------------------------------------------- base shapes
def grid_config(n):
    k = math.ceil(math.sqrt(n))
    i = np.arange(n)
    return np.column_stack([i % k + 0.5, i // k + 0.5, np.zeros(n)]), float(k)


def load_seeds():
    path = os.path.join(HERE, "seeds.json")
    seeds = {}
    if os.path.exists(path):
        for k, v in json.load(open(path)).items():
            arr = np.array(v["squares"], dtype=float)
            arr[:, 2] = np.radians(arr[:, 2])
            seeds[int(k)] = _make_seed(arr)
    # n=5: exact optimal packing (4 corner squares + 1 square tilted 45 degrees)
    s5 = 2 + math.sqrt(2) / 2
    seeds[5] = _make_seed(np.array([[0.5, 0.5, 0], [s5 - 0.5, 0.5, 0], [0.5, s5 - 0.5, 0],
                                    [s5 - 0.5, s5 - 0.5, 0], [s5 / 2, s5 / 2, math.pi / 4]]))
    return seeds


def _make_seed(cfg):
    """Strictly valid version of a base packing; side rounded UP to the exact 2**-40 grid."""
    cfg, side, _ = finalize(cfg, gap=GAP, snap_tol=1e-4, merge_deg=0.6)   # tiny tilts are optimizer noise
    return cfg, math.ceil(side / Q) * Q


# ------------------------------------------------------------------ DP search
def plan():
    seeds = load_seeds()
    side, how = {}, {}
    for n in range(1, N_MAX + 1):
        side[n] = float(math.ceil(math.sqrt(n)))
        how[n] = ("grid", None)
    for n, (_, s) in seeds.items():
        if s < side[n] - EPS:
            side[n], how[n] = s, ("optimizer", None)

    changed = True
    while changed:
        changed = False
        for n0 in range(1, N_MAX + 1):                    # L-extension
            cap = n0 + 2 * math.floor(side[n0] + 1e-12) + 1
            for n in range(n0 + 1, min(cap, N_MAX) + 1):
                if side[n0] + 1 < side[n] - EPS:
                    side[n], how[n] = side[n0] + 1, ("L-extend", n0)
                    changed = True
        for n in range(N_MAX - 1, 0, -1):                  # truncation
            if side[n + 1] < side[n] - EPS:
                side[n], how[n] = side[n + 1], ("truncate", n + 1)
                changed = True
    return side, how, seeds


def build(n, how, seeds, side, memo):
    if n in memo:
        return memo[n]
    kind, arg = how[n]
    if kind == "grid":
        cfg = grid_config(n)[0]
    elif kind == "optimizer":
        cfg = seeds[n][0]
    elif kind == "truncate":
        cfg = build(arg, how, seeds, side, memo)[:n]
    else:  # L-extend of arg
        base = build(arg, how, seeds, side, memo)
        s0 = side[arg]
        top = math.floor(s0 + 1 + 1e-12)       # squares along the new top strip
        right = math.floor(s0 + 1e-12)         # squares up the new right strip
        extra = [(i + 0.5, s0 + 0.5, 0.0) for i in range(top)]
        extra += [(s0 + 0.5, i + 0.5, 0.0) for i in range(right)]
        cfg = np.vstack([base, np.array(extra)])[:n]
    memo[n] = cfg
    return cfg


# --------------------------------------------------------------- verification
def check(cfg, side):
    """Return (wall violation, rules_ok).  Valid: wall <= EPS and rules_ok (see strict_gaps.validate)."""
    x, y, t = cfg[:, 0], cfg[:, 1], cfg[:, 2]
    h = half_extent(t)
    wall = max(np.max(h - x), np.max(h - y), np.max(x + h - side), np.max(y + h - side))
    ok, _ = validate(cfg, gap=GAP)
    return float(wall), ok


def proven_optimal(n):
    k = math.ceil(math.sqrt(n))
    return n in (1, 5, 6, 10) or n in (k * k, k * k - 1, k * k - 2)


# ----------------------------------------------------------------------- main
def main():
    side, how, seeds = plan()
    memo = {}
    final_side = {}
    rows = 0
    with open(os.path.join(HERE, "packings.csv"), "w", newline="") as f, \
         open(os.path.join(HERE, "packings_summary.csv"), "w", newline="") as g:
        w, ws = csv.writer(f, lineterminator="\n"), csv.writer(g, lineterminator="\n")
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        ws.writerow(["n", "container_side", "method", "proven_optimal_size"])
        for n in range(1, N_MAX + 1):
            cfg = text_safe(tidy_gaps(build(n, how, seeds, side, memo), tol=1e-3, gap=GAP))
            assert len(cfg) == n, (n, len(cfg))
            hh = half_extent(cfg[:, 2])      # text_safe may nudge squares by ~1e-12: recompute the side
            final_side[n] = max(side[n], float((cfg[:, 0] + hh).max()), float((cfg[:, 1] + hh).max()))
            for sid, (x, y, a) in enumerate(cfg):
                w.writerow([rows, n, sid, fmt(x), fmt(y), fmt(a)])
                rows += 1
            kind, arg = how[n]
            m = kind if arg is None else f"{kind} (from n={arg})"
            ws.writerow([n, f"{final_side[n]:.9f}", m, proven_optimal(n)])
    print("rows written:", rows)

    # ---- verify by reading the CSV back from disk
    data = {}
    with open(os.path.join(HERE, "packings.csv")) as f:
        for r in csv.DictReader(f):
            data.setdefault(int(r["n"]), []).append(
                (int(r["square_id"]), float(r["x"]), float(r["y"]), float(r["angle"])))
    bad = 0
    for n in range(1, N_MAX + 1):
        sq = sorted(data[n])
        assert [s[0] for s in sq] == list(range(n))
        wall, ok = check(np.array([s[1:] for s in sq]), final_side[n])
        if wall > EPS or not ok:
            bad += 1
            print("INVALID", n, wall, ok)
    print("verified n=1..%d, invalid: %d" % (N_MAX, bad))


if __name__ == "__main__":
    main()
