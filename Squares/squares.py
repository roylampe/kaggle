"""
Pack n unit squares (any rotation) into the smallest square container.

pack_squares(n) -> (side, [(x, y, angle_deg), ...])
  (x, y) is each square's center; the container spans [0, side] x [0, side].

Method (heuristic -- no exact algorithm is known for this problem):
  * Energy = squared penetration depth between squares (separating-axis test)
    + squared out-of-container depth.  Energy 0 means a valid packing.
  * Multi-start: random layouts are squeezed by minimizing  S + mu*energy  with
    increasing mu, then made feasible and shrunk step by step.
  * Basin hopping: relocate 1-3 random squares in the best packing, re-relax,
    shrink again, keep it if the container got smaller.  This is what lets the
    search escape grid-like local minima (e.g. n=5 stuck at side 3).
  * Runs until `time_limit` seconds are used.  More time => better results.

Requires: numpy, scipy
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

TOL = 1e-10


# ----------------------------------------------------------------------------
# Energy: 0 when squares don't overlap and all lie inside [0, S]^2
# ----------------------------------------------------------------------------
def _g(d):
    """Derivative of 0.5*(|cos d| + |sin d|) with respect to d."""
    return 0.5 * (-np.sign(np.cos(d)) * np.sin(d) + np.sign(np.sin(d)) * np.cos(d))


def _energy_grad(params, n, S, iu, ju):
    """Energy and its analytic gradient w.r.t. (x, y, theta); also dE/dS."""
    x, y, t = params[:n], params[n:2 * n], params[2 * n:]
    c, s = np.cos(t), np.sin(t)
    gx, gy, gt = np.zeros(n), np.zeros(n), np.zeros(n)

    # --- container walls ---
    h = 0.5 * (np.abs(c) + np.abs(s))
    dh = _g(t)
    a1, a2 = np.maximum(0, h - x), np.maximum(0, x + h - S)
    b1, b2 = np.maximum(0, h - y), np.maximum(0, y + h - S)
    e = np.sum(a1 ** 2 + a2 ** 2 + b1 ** 2 + b2 ** 2)
    gx += -2 * a1 + 2 * a2
    gy += -2 * b1 + 2 * b2
    gt += 2 * (a1 + a2 + b1 + b2) * dh
    dS = -2 * np.sum(a2) - 2 * np.sum(b2)

    # --- pairwise overlaps (SAT, least-penetration axis) ---
    if len(iu):
        ti, tj = t[iu], t[ju]
        dx, dy = x[ju] - x[iu], y[ju] - y[iu]
        phis = (ti, ti + np.pi / 2, tj, tj + np.pi / 2)
        ovs, info = [], []
        for phi in phis:
            di, dj = phi - ti, phi - tj
            ri = 0.5 * (np.abs(np.cos(di)) + np.abs(np.sin(di)))
            rj = 0.5 * (np.abs(np.cos(dj)) + np.abs(np.sin(dj)))
            proj = np.cos(phi) * dx + np.sin(phi) * dy
            ovs.append(ri + rj - np.abs(proj))
            info.append((di, dj, proj, phi))
        ovs = np.array(ovs)
        k = np.argmin(ovs, axis=0)
        pen = ovs[k, np.arange(len(iu))]
        act = pen > 0
        e += np.sum(pen[act] ** 2)
        if act.any():
            di = np.choose(k, [i[0] for i in info])
            dj = np.choose(k, [i[1] for i in info])
            proj = np.choose(k, [i[2] for i in info])
            phi = np.choose(k, [i[3] for i in info])
            sg = np.sign(proj)
            dq_dphi = sg * (-np.sin(phi) * dx + np.cos(phi) * dy)
            gdi, gdj = _g(di), _g(dj)
            first = k < 2  # axis tied to square i
            d_ti = np.where(first, gdj - dq_dphi, -gdi)
            d_tj = np.where(first, -gdj, gdi - dq_dphi)
            d_xj = -sg * np.cos(phi)
            d_yj = -sg * np.sin(phi)
            w = 2 * pen * act
            np.add.at(gx, ju, w * d_xj); np.add.at(gx, iu, -w * d_xj)
            np.add.at(gy, ju, w * d_yj); np.add.at(gy, iu, -w * d_yj)
            np.add.at(gt, iu, w * d_ti); np.add.at(gt, ju, w * d_tj)
    return e, np.concatenate([gx, gy, gt]), dS


def _energy(params, n, S, iu, ju):
    return _energy_grad(params, n, S, iu, ju)[0]


def _fun_grad(p, n, S, iu, ju, fix=False):
    e, g, _ = _energy_grad(p, n, S, iu, ju)
    if fix:
        g[2 * n:] = 0.0  # zero gradient => L-BFGS-B never changes the angles
    return e, g


def _relax(p, n, S, iu, ju, maxiter=400, fix=False):
    r = minimize(_fun_grad, p, args=(n, S, iu, ju, fix), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-11})
    return r.x, r.fun


def _joint(q, n, mu, iu, ju, fix=False):
    e, g, dS = _energy_grad(q[:-1], n, q[-1], iu, ju)
    if fix:
        g[2 * n:] = 0.0
    return q[-1] + mu * e, np.append(mu * g, 1.0 + mu * dS)


def _make_feasible(p, n, S, iu, ju, max_tries=40, fix=False):
    for _ in range(max_tries):
        p, e = _relax(p, n, S, iu, ju, fix=fix)
        if e < TOL:
            return S, p
        S += 0.05
    return math.inf, None


def _shrink(p, S, n, iu, ju, step=0.02, min_step=1e-4, fix=False):
    """Shrink container while a zero-energy relaxation still exists."""
    while step > min_step:
        St = S - step
        t = p.copy()
        t[:2 * n] *= St / S
        t, e = _relax(t, n, St, iu, ju, fix=fix)
        if e < TOL:
            p, S = t, St
        else:
            step *= 0.5
    return S, p


# ----------------------------------------------------------------------------
# Search moves
# ----------------------------------------------------------------------------
def _random_start(n, rng, iu, ju, angles=None):
    fix = angles is not None
    S = math.sqrt(n) * 1.15
    theta = rng.choice(angles, n) if fix else rng.uniform(0, math.pi / 2, n)
    p = np.concatenate([rng.uniform(0.5, S - 0.5, 2 * n), theta])
    q = np.append(p, S)
    for mu in (5, 30, 200, 2000, 2e4):  # squeeze, container size is a variable
        q = minimize(_joint, q, args=(n, mu, iu, ju, fix), jac=True, method="L-BFGS-B",
                     options={"maxiter": 800, "ftol": 1e-14}).x
    S, p = _make_feasible(q[:-1], n, q[-1], iu, ju, fix=fix)
    if p is None:
        return math.inf, None
    return _shrink(p, S, n, iu, ju, fix=fix)


def _hop(p, S, n, rng, iu, ju, angles=None):
    """Relocate 1-3 random squares, re-relax in a slightly larger box, shrink."""
    q = p.copy()
    idx = rng.choice(n, rng.integers(1, min(n, 3) + 1), replace=False)
    Sg = S + 0.35
    q[:2 * n] *= Sg / S
    q[idx] = rng.uniform(0.5, Sg - 0.5, len(idx))
    q[n + idx] = rng.uniform(0.5, Sg - 0.5, len(idx))
    fix = angles is not None
    q[2 * n + idx] = (rng.choice(angles, len(idx)) if fix
                      else rng.uniform(0, math.pi / 2, len(idx)))
    q[:2 * n] += rng.normal(0, 0.03, 2 * n)
    Sg, q = _make_feasible(q, n, Sg, iu, ju, max_tries=20, fix=fix)
    if q is None:
        return math.inf, None
    return _shrink(q, Sg, n, iu, ju, step=0.02, min_step=2e-3, fix=fix)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def _snap_angles(p, n, S, iu, ju, tol_deg=3.0):
    """
    Snap angles lying within `tol_deg` of a multiple of 45 degrees to that exact
    value (0 or 45), keeping each change only if positions can be re-relaxed to a
    valid packing.  Removes noise such as 2e-07 rad and drift such as 44.4 deg.
    """
    p = p.copy()
    for i in range(n):
        deg = math.degrees(p[2 * n + i]) % 90.0
        target = round(deg / 45.0) * 45.0
        if abs(deg - target) > tol_deg:
            continue
        q = p.copy()
        q[2 * n + i] = math.radians(target)
        q, e = _relax(q, n, S, iu, ju, fix=True)   # positions only, angles frozen
        if e < TOL:
            p = q
            p[2 * n + i] = math.radians(target)    # exact value, not re-relaxed
    return p


def max_violation(side, squares):
    """Largest overlap / out-of-bounds depth (~0 for a valid packing)."""
    n = len(squares)
    p = np.array([s[0] for s in squares] + [s[1] for s in squares] +
                 [math.radians(s[2]) for s in squares])
    iu, ju = np.triu_indices(n, k=1)
    return math.sqrt(_energy(p, n, side, iu, ju))


def plot_packing(side, squares, filename="packing.png"):
    """Draw the container and every unit square (labelled with its number and angle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(Rectangle((0, 0), side, side, fill=False, lw=2.5, ec="black"))
    cmap = plt.get_cmap("tab10")
    for i, (x, y, a) in enumerate(squares):
        t = math.radians(a)
        c, s = math.cos(t), math.sin(t)
        corners = [(x + c * dx - s * dy, y + s * dx + c * dy)
                   for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]
        ax.add_patch(Polygon(corners, closed=True, fc=cmap(i % 10), ec="black", alpha=0.75))
        ax.text(x, y, f"{i + 1}\n{a:.1f}\u00b0", ha="center", va="center", fontsize=9)
    ax.set_xlim(-0.1, side + 0.1)
    ax.set_ylim(-0.1, side + 0.1)
    ax.set_aspect("equal")
    ax.set_title(f"{len(squares)} unit squares in a {side:.4f} x {side:.4f} container")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def pack_squares(n, time_limit=60.0, seed=0, verbose=False, angles=None, snap=True,
                 return_stats=False):
    """
    Find a small square container holding n unit squares.

    Parameters
    ----------
    n          : number of unit squares
    time_limit : seconds to search (more time => better results)
    seed       : RNG seed
    angles     : optional list of allowed rotations in degrees, e.g. [0, 45].
                 If given, every square is restricted to one of these angles
                 (the solver chooses which square gets which angle and where it
                 goes).  If None, angles are free to take any value.
    snap       : if True (default) and angles are free, angles within 3 degrees
                 of 0/45 are snapped to exactly 0/45 when the packing stays valid
    return_stats : if True, return (side, squares, stats) where stats is a dict:
                 attempts      total candidate packings attempted (random starts + hops)
                 random_starts fresh random layouts tried
                 hops          basin-hopping moves tried (relocating 1-3 squares)
                 improvements  attempts that found a smaller container
                 seconds       time spent searching

    Returns
    -------
    side    : container side length (an upper bound on the true optimum)
    squares : list of (x, y, angle_deg); angle in [0, 90)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        res = (1.0, [(0.5, 0.5, 0.0)])
        stats = {"attempts": 0, "random_starts": 0, "hops": 0, "improvements": 0, "seconds": 0.0}
        return res + (stats,) if return_stats else res

    fix = angles is not None
    if fix:
        angles = sorted({round(float(a) % 90.0, 9) for a in angles})
        if not angles:
            raise ValueError("angles must contain at least one value")
        angles_rad = np.radians(angles)
    else:
        angles_rad = None

    t0 = time.time()
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    best_S, best_p = math.inf, None
    stall = 0
    n_starts = n_hops = n_improved = 0

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    cur_S, cur_p = math.inf, None
    while time.time() - t0 < time_limit:
        if cur_p is None or stall >= 15:        # (re)start
            cur_S, cur_p = _random_start(n, rng, iu, ju, angles_rad)
            n_starts += 1
            stall = 0
            log(f"new random start -> {cur_S:.5f}")
        else:                                    # basin hop
            S2, p2 = _hop(cur_p, cur_S, n, rng, iu, ju, angles_rad)
            n_hops += 1
            if S2 < cur_S - 1e-6:
                cur_S, cur_p, stall = S2, p2, 0
            else:
                stall += 1
        if cur_p is not None and cur_S < best_S - 1e-9:
            best_S, best_p = cur_S, cur_p.copy()
            n_improved += 1
            log(f"best side = {best_S:.5f}")

    # final fine polish
    S, p = _shrink(best_p, best_S, n, iu, ju, step=1e-2, min_step=1e-7, fix=fix)
    side = S + 1e-5  # tiny margin to absorb residual numerical overlap
    p, _ = _relax(p, n, side, iu, ju, fix=fix)
    if snap and not fix:
        p = _snap_angles(p, n, side, iu, ju)
    # Scale centres up by 1 part in a million so that squares which merely touch
    # end up strictly non-overlapping (removes ~1e-9 numerical overlaps).
    f = 1.0 + 1e-6
    p[:2 * n] *= f
    side *= f
    x, y, t = p[:n], p[n:2 * n], p[2 * n:]
    ang = np.round(np.degrees(t), 9) % 90.0
    res = (float(side), [(float(x[i]), float(y[i]), float(ang[i])) for i in range(n)])
    if not return_stats:
        return res
    stats = {"attempts": n_starts + n_hops, "random_starts": n_starts, "hops": n_hops,
             "improvements": n_improved, "seconds": time.time() - t0}
    return res + (stats,)


def pack_many(ns, time_limit=60.0, seed=0, verbose=False, angles=None, return_stats=False):
    """
    Run pack_squares for every n in `ns`.
    Returns {n: (side, squares)}, or ({n: (side, squares)}, {n: stats}) if
    return_stats=True (stats as described in pack_squares).
    """
    results, all_stats = {}, {}
    for n in ns:
        side, squares, stats = pack_squares(n, time_limit=time_limit, seed=seed,
                                            verbose=verbose, angles=angles, return_stats=True)
        results[n], all_stats[n] = (side, squares), stats
    return (results, all_stats) if return_stats else results


def save_csv(results, filename="packings.csv", append=False, clockwise=False, start_id=None):
    """
    Write packings to a CSV with columns:  id, n, square_id, x, y, angle

      id         row identifier (0 .. number of rows - 1)
      n          which puzzle the square belongs to
      square_id  index of the square within its puzzle (0 .. n-1)
      x, y       centre of the square (container is [0, side] x [0, side])
      angle      rotation in RADIANS

    `results` is either {n: (side, squares)} as returned by pack_many(), or a
    single (side, squares) tuple as returned by pack_squares().
    Angles are counter-clockwise (standard maths convention, y axis pointing up)
    and lie in [0, pi/2) because a unit square looks the same after a quarter turn.
    Pass clockwise=True if your checker measures rotation clockwise.
    start_id sets the `id` of the first row written.  If None (default), ids start
    at 0, or continue from the last row of the file when append=True.
    With append=True, rows are added to an existing file and `id` continues from
    its last row (the header is only written for a new/empty file).
    Returns the number of rows written.
    """
    import csv
    import os

    if isinstance(results, tuple):  # single (side, squares)
        results = {len(results[1]): results}

    next_id, write_header = 0, True
    explicit_start = start_id is not None
    if append and os.path.exists(filename) and os.path.getsize(filename) > 0:
        write_header = False
        with open(filename, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            next_id = int(rows[-1][0]) + 1
    if explicit_start:
        next_id = int(start_id)

    count = 0
    with open(filename, "a" if append else "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if write_header:
            w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for n in sorted(results):
            _, squares = results[n]
            for sid, (x, y, a_deg) in enumerate(squares):
                if clockwise:
                    a_deg = (-a_deg) % 90.0
                if a_deg < 1e-9 or a_deg > 90.0 - 1e-9:
                    a_deg = 0.0            # avoid 1e-10 / 89.9999999 noise
                w.writerow([next_id, n, sid, f"{x:.15g}", f"{y:.15g}",
                            f"{math.radians(a_deg):.15g}"])
                next_id += 1
                count += 1
    return count


def _parse_ns(spec):
    """'5' -> [5];  '1-10' -> [1..10];  '1,3,7-9' -> [1,3,7,8,9]"""
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pack n unit squares in the smallest square.")
    ap.add_argument("n", nargs="?", default="5",
                    help='number of squares: "5", a range "1-10", or a list "1,3,7-9"')
    ap.add_argument("time", nargs="?", type=float, default=60,
                    help="seconds of search PER n (default 60)")
    ap.add_argument("angles", nargs="?", default=None,
                    help='optional allowed angles in degrees, e.g. "0,45"')
    ap.add_argument("--csv", metavar="FILE", help="write results to a CSV file "
                    "(columns: id,n,square_id,x,y,angle[radians])")
    ap.add_argument("--append", action="store_true", help="append to the CSV instead of overwriting")
    ap.add_argument("--start-id", type=int, default=None, metavar="ID",
                    help="row id of the first CSV row (default: 0, or continue after the "
                    "last row when appending)")
    ap.add_argument("--clockwise", action="store_true",
                    help="write angles as clockwise rotations (default: counter-clockwise)")
    ap.add_argument("--no-plot", action="store_true", help="don't save a PNG (single n only)")
    args = ap.parse_args()

    ns = _parse_ns(args.n)
    angs = [float(a) for a in args.angles.split(",")] if args.angles else None
    results, total_attempts = {}, 0
    for n in ns:
        side, squares, st = pack_squares(n, time_limit=args.time, verbose=len(ns) == 1,
                                         angles=angs, return_stats=True)
        results[n] = (side, squares)
        total_attempts += st["attempts"]
        print(f"n = {n}: container side ~= {side:.5f}   "
              f"max violation {max_violation(side, squares):.1e}   "
              f"combinations attempted: {st['attempts']:,} "
              f"({st['random_starts']:,} random starts + {st['hops']:,} hops)", flush=True)
        if args.csv:  # save progress after each n so long runs are never lost
            first = len(results) == 1   # --start-id applies to the first row only
            save_csv({n: results[n]}, args.csv, append=args.append or not first,
                     clockwise=args.clockwise, start_id=args.start_id if first else None)

    if len(ns) == 1:
        side, squares = results[ns[0]]
        if not args.no_plot:
            try:
                print("image saved to", plot_packing(side, squares, f"packing_n{ns[0]}.png"))
            except ImportError:
                print("(install matplotlib to save an image)")
        for i, (x, y, a) in enumerate(squares, 1):
            print(f"  square {i:2d}: center = ({x:8.4f}, {y:8.4f})  angle = {a:7.3f} deg")
    print(f"\nTotal combinations attempted: {total_attempts:,}")
    if args.csv:
        print("CSV written to", args.csv)