"""
Pack n unit squares (any rotation) into the smallest square container.

pack_squares(n) -> (side, [(x, y, angle_deg), ...])
  (x, y) is each square's center; the container spans [0, side] x [0, side].

Method (heuristic -- no exact algorithm is known for this problem):
  * Energy = squared penetration depth between squares (separating-axis test)
    + squared out-of-container depth.  Energy 0 means a valid packing.
  * Seeding: the search starts from the best constructed layout (grid, exact n=5,
    L-extensions of smaller packings, truncation); half of the restarts reuse it.
  * Multi-start: random layouts are squeezed by minimizing  S + mu*energy  with
    increasing mu, then made feasible and shrunk step by step.
  * Basin hopping: relocate 1-3 random squares in the best packing, re-relax,
    shrink again, keep it if the container got smaller.  This is what lets the
    search escape grid-like local minima (e.g. n=5 stuck at side 3).
  * Runs until `time_limit` seconds are used.  More time => better results.

Requires: numpy, scipy   (matplotlib optional, only for plot_packing)

=============================== USAGE EXAMPLES ===============================

PYTHON
------
    from square_packing import (pack_squares, pack_many, save_csv,
                                plot_packing, max_violation)

    # 1. Basic: 5 squares, search for 30 seconds
    side, squares = pack_squares(5, time_limit=30)
    #   side    -> 2.7071...
    #   squares -> [(x, y, angle_degrees), ...]   one tuple per square

    # 2. Reproducible run / different random search (more seeds = more chances)
    best = min((pack_squares(11, time_limit=30, seed=s) for s in range(5)),
               key=lambda r: r[0])

    # 3. Restrict the allowed rotations (degrees), e.g. only 0 or 45
    side, squares = pack_squares(10, time_limit=60, angles=[0, 45])

    # 4. Search statistics: how many combinations were attempted
    side, squares, stats = pack_squares(5, time_limit=20, return_stats=True)
    print(stats["attempts"], stats["random_starts"], stats["seeded_starts"],
          stats["hops"], stats["improvements"], stats["seconds"])

    # 5. Pure random starts (turn the constructed seed off)
    side, squares = pack_squares(17, time_limit=60, construct=False)

    # 6. Reuse earlier results so larger n can build on them (L-extension).
    #    pack_many does this automatically when n is ascending.
    results = pack_many(range(1, 21), time_limit=30)          # {n: (side, squares)}
    results, stats = pack_many([11, 18], time_limit=45, return_stats=True)
    #    or by hand:
    side11, sq11 = pack_squares(11, time_limit=45)
    side18, sq18 = pack_squares(18, time_limit=45, bases={11: (side11, sq11)})

    # 7. Search tuning (defaults shown): see "tuning" in pack_squares docstring
    pack_squares(11, time_limit=60, stall_limit=15, max_hop=3, hop_margin=0.35)

    # 8. Keep angles exactly as found instead of snapping to 0/45 degrees
    pack_squares(5, time_limit=20, snap=False)

    # 8b. Best-known side length given: search a fixed container of that size, stop when reached
    side, squares, stats = pack_squares(11, time_limit=200, target=3.877084, return_stats=True)
    print(stats["reached_target"])             # True / False
    #     several seeds in parallel (uses all CPU cores, stops when one seed reaches the target)
    side, squares, stats = pack_best_of(11, seeds=8, time_limit=200, target=3.877084)
    #     targets for many n at once, from a CSV with columns n, best_known
    results = pack_many(range(1, 31), time_limit=60, targets=read_targets("best_known.csv"))

    # 9. Verify a result (largest overlap/out-of-bounds depth, ~0 if valid)
    print(max_violation(side, squares))

    # 10. Save to CSV: columns  id, n, square_id, x, y, angle (RADIANS)
    save_csv((side, squares), "one.csv")                      # single packing
    save_csv(results, "all.csv")                              # many n in one file
    save_csv(results, "all.csv", append=True)                 # add rows to a file
    save_csv(results, "all.csv", start_id=1000)               # first row id = 1000
    save_csv(results, "all.csv", clockwise=True)              # clockwise angles

    # 11. Draw it (needs matplotlib)
    plot_packing(side, squares, "packing.png")

COMMAND LINE      python square_packing.py N [SECONDS] [ANGLES] [options]
------------
    python square_packing.py 5                       # n=5, 60 s, saves packing_n5.png
    python square_packing.py 5 30                    # n=5, 30 s
    python square_packing.py 10 60 0,45              # only 0 and 45 degree squares
    python square_packing.py 1-10 20 --csv out.csv   # n = 1..10, 20 s each, CSV output
    python square_packing.py 1,3,7-9 20 --csv out.csv        # a list/ranges of n
    python square_packing.py 11-12 30 --csv out.csv --append # add to an existing CSV
    python square_packing.py 5 30 --csv out.csv --start-id 100   # first id = 100
    python square_packing.py 5 30 --csv out.csv --clockwise      # clockwise angles
    python square_packing.py 17 60 --no-construct    # random starts only
    python square_packing.py 11 60 --seed 3          # different random seed
    python square_packing.py 11 60 --stall-limit 5 --max-hop 2 --hop-margin 0.6
    python square_packing.py 5 30 --no-snap --no-plot
    python square_packing.py 11 200 --target 3.877084          # best-known side given
    python square_packing.py 11 200 --target 3.877084 --seeds 8 --workers 4
    python square_packing.py 1-30 60 --targets best_known.csv --csv out.csv
    python square_packing.py --help                  # all options
==============================================================================
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

    # --- pairwise overlaps (SAT, least-penetration axis), vectorised over the 4 axes ---
    if len(iu):
        ti, tj = t[iu], t[ju]
        dx, dy = x[ju] - x[iu], y[ju] - y[iu]
        phis = np.stack((ti, ti + np.pi / 2, tj, tj + np.pi / 2))        # (4, pairs)
        di, dj = phis - ti, phis - tj
        ri = 0.5 * (np.abs(np.cos(di)) + np.abs(np.sin(di)))
        rj = 0.5 * (np.abs(np.cos(dj)) + np.abs(np.sin(dj)))
        proj = np.cos(phis) * dx + np.sin(phis) * dy
        ovs = ri + rj - np.abs(proj)
        k = ovs.argmin(axis=0)
        pen = ovs[k, np.arange(len(iu))]
        act = np.flatnonzero(pen > 0)
        if act.size:
            ka = k[act]
            pa = pen[act]
            e += np.sum(pa ** 2)
            dia, dja = di[ka, act], dj[ka, act]
            pj, phi = proj[ka, act], phis[ka, act]
            sg = np.sign(pj)
            dq_dphi = sg * (-np.sin(phi) * dx[act] + np.cos(phi) * dy[act])
            gdi, gdj = _g(dia), _g(dja)
            first = ka < 2                                   # axis tied to square i
            d_ti = np.where(first, gdj - dq_dphi, -gdi)
            d_tj = np.where(first, -gdj, gdi - dq_dphi)
            w = 2 * pa
            wx, wy = w * (-sg * np.cos(phi)), w * (-sg * np.sin(phi))
            ia, ja = iu[act], ju[act]
            gx += np.bincount(ja, wx, n) - np.bincount(ia, wx, n)
            gy += np.bincount(ja, wy, n) - np.bincount(ia, wy, n)
            gt += np.bincount(ia, w * d_ti, n) + np.bincount(ja, w * d_tj, n)
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


def _construction(n, bases=None):
    """
    Best layout obtainable *without searching*, from:
      grid            ceil(sqrt(m)) x ceil(sqrt(m)) axis-aligned grid
      exact m=5       4 corner squares + one tilted 45 degrees (side 2 + 1/sqrt 2)
      L-extension     a packing of m0 squares in side s0 plus an "L" of axis-aligned
                      squares along two sides -> side s0 + 1 holds m0 + 2*floor(s0) + 1
      truncation      drop squares from a packing of more squares
      bases           optional {m: (side, squares_deg)} of earlier results
    Returns (side, p) with p = [x..., y..., theta_radians...], or (inf, None).
    """
    N = n + 3 * math.ceil(math.sqrt(n)) + 3
    side = {m: float(math.ceil(math.sqrt(m))) for m in range(1, N + 1)}
    how = {m: ("grid", None) for m in range(1, N + 1)}
    s5 = 2 + math.sqrt(2) / 2
    if N >= 5 and s5 < side[5]:
        side[5], how[5] = s5, ("five", None)
    for m, (sd, _) in (bases or {}).items():
        if m <= N and sd < side[m] - 1e-9:
            side[m], how[m] = sd, ("base", None)

    changed = True
    while changed:
        changed = False
        for m0 in range(1, N + 1):                      # L-extension
            cap = m0 + 2 * math.floor(side[m0] + 1e-12) + 1
            for m in range(m0 + 1, min(cap, N) + 1):
                if side[m0] + 1 < side[m] - 1e-9:
                    side[m], how[m] = side[m0] + 1, ("L", m0)
                    changed = True
        for m in range(N - 1, 0, -1):                   # truncation
            if side[m + 1] < side[m] - 1e-9:
                side[m], how[m] = side[m + 1], ("trunc", m + 1)
                changed = True

    memo = {}

    def build(m):
        if m in memo:
            return memo[m]
        kind, arg = how[m]
        if kind == "grid":
            k = math.ceil(math.sqrt(m))
            i = np.arange(m)
            cfg = np.column_stack([i % k + 0.5, i // k + 0.5, np.zeros(m)])
        elif kind == "five":
            cfg = np.array([[0.5, 0.5, 0], [s5 - 0.5, 0.5, 0], [0.5, s5 - 0.5, 0],
                            [s5 - 0.5, s5 - 0.5, 0], [s5 / 2, s5 / 2, math.pi / 4]])
        elif kind == "base":
            cfg = np.array(bases[m][1], dtype=float)
            cfg[:, 2] = np.radians(cfg[:, 2])
        elif kind == "trunc":
            cfg = build(arg)[:m]
        else:  # L
            s0 = side[arg]
            top, right = math.floor(s0 + 1 + 1e-12), math.floor(s0 + 1e-12)
            extra = [(i + 0.5, s0 + 0.5, 0.0) for i in range(top)]
            extra += [(s0 + 0.5, i + 0.5, 0.0) for i in range(right)]
            cfg = np.vstack([build(arg), np.array(extra)])[:m]
        memo[m] = cfg
        return cfg

    cfg = build(n)
    return side[n], np.concatenate([cfg[:, 0], cfg[:, 1], cfg[:, 2]])


def _feasible_search(n, S, rng, iu, ju, deadline, angles=None, max_hop=3, stall_limit=40,
                     hop_maxiter=400, counter=None):
    """
    Look for a packing of n unit squares in a container of FIXED side S (energy = 0).
    Energy-based hopping: relocate the squares that overlap most (chosen with probability
    proportional to their overlap gradient), re-relax, keep the move if the energy drops;
    restart from a random layout when stuck.  Returns the packing vector, or None on timeout.
    """
    fix = angles is not None

    def new_angles(k):
        return rng.choice(angles, k) if fix else rng.uniform(0, math.pi / 2, k)

    def start():
        p = np.concatenate([rng.uniform(0.5, S - 0.5, 2 * n), new_angles(n)])
        if counter is not None:
            counter[0] += 1
        return _relax(p, n, S, iu, ju, fix=fix)

    p, e = start()
    stall = 0
    while time.time() < deadline:
        if e < TOL:
            return p
        if stall >= stall_limit:
            p, e = start()
            stall = 0
            continue
        _, g, _ = _energy_grad(p, n, S, iu, ju)
        w = np.hypot(g[:n], g[n:2 * n]) + 1e-12
        w = w / w.sum()
        if rng.random() < 0.3:                      # sometimes pick squares uniformly
            w = np.full(n, 1.0 / n)
        k = int(rng.integers(1, min(n, max_hop) + 1))
        idx = rng.choice(n, k, replace=False, p=w)
        q = p.copy()
        q[idx] = rng.uniform(0.5, S - 0.5, k)
        q[n + idx] = rng.uniform(0.5, S - 0.5, k)
        q[2 * n + idx] = new_angles(k)
        q[:2 * n] += rng.normal(0, 0.01, 2 * n)
        if counter is not None:
            counter[0] += 1
        q, eq = _relax(q, n, S, iu, ju, maxiter=hop_maxiter, fix=fix)
        if eq < e - 1e-12:
            p, e, stall = q, eq, 0
        else:
            stall += 1
    return p if e < TOL else None


def _hop(p, S, n, rng, iu, ju, angles=None, max_hop=3, hop_margin=0.35):
    """Relocate 1-3 random squares, re-relax in a slightly larger box, shrink."""
    q = p.copy()
    idx = rng.choice(n, rng.integers(1, min(n, max_hop) + 1), replace=False)
    Sg = S + hop_margin
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
                 return_stats=False, stall_limit=15, max_hop=3, hop_margin=0.35,
                 construct=True, bases=None, target=None, target_slack=None, target_tol=1e-4,
                 target_fraction=0.8):
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
    construct  : if True (default, ignored when `angles` is given) the search is seeded
                 with the best constructed layout (grid / L-extension / truncation,
                 see _construction) and half of the restarts begin from it instead of
                 a random layout.  Set False for purely random starts.
    bases      : optional {m: (side, squares)} of earlier results (e.g. from pack_many)
                 that the construction may extend with L-layers.
    target     : optional best-known side length for this n.  The search then (1) looks for
                 a packing in a container of exactly target + target_slack (a pure
                 feasibility search, see _feasible_search) using up to target_fraction of the
                 time, tightens it, and (2) stops as soon as side <= target + target_tol.
                 A wrong (too small) target is harmless: the search falls back to normal.
    target_slack : room added to the target for the feasibility search (default 0.13% of the
                 target, e.g. 0.005 for n=11).  The optimum itself fits exactly, but a container
                 that tight is a needle in a haystack (2e-4 failed in every test); with ~0.13%
                 the search finds the best-known layout and the shrink step then tightens it
    target_tol : stop early when side <= target + target_tol (default 1e-4)
    target_fraction : share of time_limit spent on the feasibility search (default 0.8)
    stall_limit : failed hops in a row before abandoning the current layout and
                 starting a fresh random one (default 15)
    max_hop    : a hop relocates 1..max_hop random squares (default 3)
    hop_margin : extra container size given to a hop so it can re-pack (default 0.35)
    return_stats : if True, return (side, squares, stats) where stats is a dict:
                 attempts      total candidate packings attempted (random starts + hops)
                 random_starts fresh random layouts tried
                 seeded_starts starts from the constructed layout (see `construct`)
                 fixed_container_attempts  relaxations in the fixed-container search (target)
                 hops          basin-hopping moves tried (relocating 1-3 squares)
                 improvements  attempts that found a smaller container
                 reached_target True if side <= target + target_tol (None if no target given)
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
        stats = {"attempts": 0, "random_starts": 0, "seeded_starts": 0, "fixed_container_attempts": 0, "hops": 0,
                 "improvements": 0, "seconds": 0.0, "reached_target": None if target is None else True}
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
    n_starts = n_hops = n_improved = n_seeded = n_feas = 0
    use_construct = construct and not fix

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    cur_S, cur_p = math.inf, None
    if target is not None:                       # fixed-container feasibility search
        if target_slack is None:
            target_slack = 0.0013 * float(target)
        S_t = float(target) + target_slack
        feas_count = [0]
        p_f = _feasible_search(n, S_t, rng, iu, ju, t0 + time_limit * target_fraction,
                               angles_rad, max_hop, counter=feas_count)
        n_feas = feas_count[0]
        if p_f is not None:
            cur_S, cur_p = _shrink(p_f, S_t, n, iu, ju, step=target_slack, fix=fix)
            best_S, best_p = cur_S, cur_p.copy()
            n_improved += 1
            log(f"feasibility search found side {best_S:.6f} (target {target:.6f})")
        else:
            log(f"no packing found in a container of side {S_t:.6f}; falling back to normal search")
    while time.time() - t0 < time_limit:
        if target is not None and best_S <= target + target_tol:
            break
        if cur_p is None or stall >= stall_limit:        # (re)start
            if use_construct and (cur_p is None or rng.random() < 0.5):
                S0, p0 = _construction(n, bases)
                cur_S, cur_p = _shrink(p0, S0, n, iu, ju)
                n_seeded += 1
                log(f"seeded start (construction) -> {cur_S:.5f}")
            else:
                cur_S, cur_p = _random_start(n, rng, iu, ju, angles_rad)
                n_starts += 1
                log(f"new random start -> {cur_S:.5f}")
            stall = 0
        else:                                    # basin hop
            S2, p2 = _hop(cur_p, cur_S, n, rng, iu, ju, angles_rad, max_hop, hop_margin)
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
    # Make the packing strictly valid: squares with different angles get a gap of 1e-6,
    # squares with the same angle stay exactly flush (see strict_gaps.py).
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from strict_gaps import finalize
        cfg, side, _ = finalize(np.column_stack([p[:n], p[n:2 * n], p[2 * n:]]),
                                gap=1e-6, snap_tol=1e-4)
        x, y, t = cfg[:, 0], cfg[:, 1], cfg[:, 2]
    except ImportError:                       # strict_gaps.py missing: old behaviour
        f = 1.0 + 1e-6
        p[:2 * n] *= f
        side *= f
        x, y, t = p[:n], p[n:2 * n], p[2 * n:]
    ang = np.round(np.degrees(t), 9) % 90.0
    res = (float(side), [(float(x[i]), float(y[i]), float(ang[i])) for i in range(n)])
    if not return_stats:
        return res
    stats = {"attempts": n_starts + n_seeded + n_hops + n_feas, "random_starts": n_starts,
             "fixed_container_attempts": n_feas,
             "seeded_starts": n_seeded, "hops": n_hops,
             "improvements": n_improved, "seconds": time.time() - t0,
             "reached_target": None if target is None else bool(best_S <= target + target_tol)}
    return res + (stats,)


def pack_many(ns, time_limit=60.0, seed=0, verbose=False, angles=None, return_stats=False,
              targets=None, **kwargs):
    """
    Run pack_squares for every n in `ns`.
    Returns {n: (side, squares)}, or ({n: (side, squares)}, {n: stats}) if
    return_stats=True (stats as described in pack_squares).
    targets: optional {n: best_known_side}; each n is searched with its own target.
    Extra keyword arguments are passed on to pack_squares.
    """
    results, all_stats = {}, {}
    for n in ns:
        side, squares, stats = pack_squares(n, time_limit=time_limit, seed=seed,
                                            verbose=verbose, angles=angles, return_stats=True,
                                            bases=dict(results),
                                            target=(targets or {}).get(n), **kwargs)
        results[n], all_stats[n] = (side, squares), stats
    return (results, all_stats) if return_stats else results


def _run_one(job):
    """Worker for pack_best_of (module level so it can be pickled)."""
    n, kwargs = job
    return pack_squares(n, return_stats=True, **kwargs)


def pack_best_of(n, seeds=4, workers=None, time_limit=60.0, seed=0, **kwargs):
    """
    Run pack_squares for `seeds` different random seeds in parallel processes and keep the
    smallest container.  With target=... the other runs are stopped as soon as one seed
    reaches it (the time to find a best-known layout is random, so several seeds in parallel
    are the most effective way to use several CPU cores).
    Returns (side, squares, stats) where stats comes from the winning run and also lists
    "seeds_run" and "side_per_seed".
    """
    import multiprocessing as mp
    import os
    workers = workers or min(seeds, os.cpu_count() or 1)
    jobs = [(n, dict(kwargs, time_limit=time_limit, seed=seed + k)) for k in range(seeds)]
    target = kwargs.get("target")
    best, sides = None, []
    if workers <= 1:
        for job in jobs:
            r = _run_one(job)
            sides.append(r[0])
            best = r if best is None or r[0] < best[0] else best
            if target is not None and r[2]["reached_target"]:
                break
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(_run_one, jobs):
                sides.append(r[0])
                best = r if best is None or r[0] < best[0] else best
                if target is not None and r[2]["reached_target"]:
                    pool.terminate()                         # a seed hit the target: stop the rest
                    break
    side, squares, stats = best
    stats = dict(stats, seeds_run=len(sides), side_per_seed=sides)
    return side, squares, stats


def read_targets(path):
    """
    Read best-known side lengths from a CSV: a column `n` and one of
    best_known / best_known_side / side / container_side / L / best (or simply the first two
    columns, with or without a header).  Returns {n: side}.
    """
    import csv
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    header = [c.strip().lower() for c in rows[0]]
    try:
        float(rows[0][0])
        has_header = False
    except ValueError:
        has_header = True
    if not has_header:
        return {int(float(r[0])): float(r[1]) for r in rows if len(r) >= 2}
    ni = header.index("n") if "n" in header else 0
    names = ("best_known", "best_known_side", "side", "container_side", "l", "best", "length")
    vi = next((header.index(c) for c in names if c in header),
              next(k for k in range(len(header)) if k != ni))
    return {int(float(r[ni])): float(r[vi]) for r in rows[1:] if len(r) > max(ni, vi) and r[vi].strip()}


def _fmt(v):
    """Shortest text that reads back as exactly the same float (0.0 -> '0', 1.5 -> '1.5')."""
    v = float(v)
    if v == 0.0:
        return "0"
    r = repr(v)
    return r[:-2] if r.endswith(".0") else r


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
                w.writerow([next_id, n, sid, _fmt(x), _fmt(y), _fmt(math.radians(a_deg))])
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

    ap = argparse.ArgumentParser(
        description="Pack n unit squares in the smallest square.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python square_packing.py 5                        n=5, 60 s, saves packing_n5.png
  python square_packing.py 5 30                     n=5, 30 s
  python square_packing.py 10 60 0,45               only 0 and 45 degree squares
  python square_packing.py 1-10 20 --csv out.csv    n=1..10, 20 s each, write CSV
  python square_packing.py 1,3,7-9 20 --csv out.csv a list / ranges of n
  python square_packing.py 11-12 30 --csv out.csv --append
                                                    add rows to an existing CSV
  python square_packing.py 5 30 --csv out.csv --start-id 100
                                                    first CSV row id = 100
  python square_packing.py 5 30 --csv out.csv --clockwise
                                                    write clockwise angles
  python square_packing.py 17 60 --no-construct     random starts only (no seeding)
  python square_packing.py 11 60 --seed 3           different random seed
  python square_packing.py 11 60 --stall-limit 5 --max-hop 2 --hop-margin 0.6
                                                    tune the search
  python square_packing.py 5 30 --no-snap --no-plot keep raw angles, skip the PNG
  python square_packing.py 11 200 --target 3.877084
                                                    best-known side given: fixed-container
                                                    search + stop when reached
  python square_packing.py 11 200 --target 3.877084 --seeds 8 --workers 4
                                                    8 seeds in parallel on 4 cores
  python square_packing.py 1-30 60 --targets best_known.csv --csv out.csv
                                                    targets for many n (CSV: n, best_known)

When n is a range (e.g. 1-30) each result is reused as a building block for the
larger n that follow (L-extension seeding), so list n in ascending order.
""")
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
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    ap.add_argument("--no-construct", action="store_true",
                    help="don't seed the search with constructed layouts (random starts only)")
    ap.add_argument("--no-snap", action="store_true",
                    help="don't snap angles near 0/45 degrees to exact values")
    ap.add_argument("--stall-limit", type=int, default=15, metavar="K",
                    help="failed hops in a row before a fresh start (default 15)")
    ap.add_argument("--max-hop", type=int, default=3, metavar="M",
                    help="a hop relocates 1..M squares (default 3)")
    ap.add_argument("--hop-margin", type=float, default=0.35, metavar="X",
                    help="extra container size given to a hop (default 0.35)")
    ap.add_argument("--target", type=float, default=None, metavar="SIDE",
                    help="best-known side length (single n): search a container of that size and "
                         "stop when reached")
    ap.add_argument("--targets", metavar="FILE",
                    help="CSV of best-known side lengths (columns n and best_known/side/...) for "
                         "several n")
    ap.add_argument("--target-slack", type=float, default=None, metavar="X",
                    help="room added to the target for the fixed-container search "
                         "(default 0.13%% of the target)")
    ap.add_argument("--target-fraction", type=float, default=0.8, metavar="F",
                    help="share of the time spent on the fixed-container search (default 0.8)")
    ap.add_argument("--seeds", type=int, default=1, metavar="K",
                    help="run K random seeds (seed, seed+1, ...) and keep the best; with "
                         "--workers they run in parallel (default 1)")
    ap.add_argument("--workers", type=int, default=None, metavar="W",
                    help="parallel processes for --seeds (default: number of CPU cores)")
    args = ap.parse_args()

    ns = _parse_ns(args.n)
    targets = read_targets(args.targets) if args.targets else {}
    if args.target is not None:
        targets = {n: args.target for n in ns}
    angs = [float(a) for a in args.angles.split(",")] if args.angles else None
    results, total_attempts = {}, 0
    for n in ns:
        kw = dict(time_limit=args.time, seed=args.seed, angles=angs, snap=not args.no_snap,
                  construct=not args.no_construct, stall_limit=args.stall_limit,
                  max_hop=args.max_hop, hop_margin=args.hop_margin, bases=dict(results),
                  target=targets.get(n), target_slack=args.target_slack,
                  target_fraction=args.target_fraction)
        if args.seeds > 1:
            side, squares, st = pack_best_of(n, seeds=args.seeds, workers=args.workers, **kw)
            print(f"  {st['seeds_run']} seed(s) run; sides: "
                  + ", ".join(f"{v:.5f}" for v in st['side_per_seed']), flush=True)
        else:
            kw.pop("time_limit"); kw.pop("seed")
            side, squares, st = pack_squares(n, time_limit=args.time, seed=args.seed,
                                             verbose=len(ns) == 1, return_stats=True, **kw)
        results[n] = (side, squares)
        total_attempts += st["attempts"]
        tgt = ("" if targets.get(n) is None else
               f"   target {targets[n]:.5f} {'REACHED' if st['reached_target'] else 'not reached'}")
        print(f"n = {n}: container side ~= {side:.5f}{tgt}   "
              f"max violation {max_violation(side, squares):.1e}   "
              f"combinations attempted: {st['attempts']:,} "
              f"({st['random_starts']:,} random + {st['seeded_starts']:,} seeded starts + {st['hops']:,} hops"
              f" + {st['fixed_container_attempts']:,} fixed-container)", flush=True)
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
