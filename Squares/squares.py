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


def _fun_grad(p, n, S, iu, ju):
    e, g, _ = _energy_grad(p, n, S, iu, ju)
    return e, g


def _relax(p, n, S, iu, ju, maxiter=400):
    r = minimize(_fun_grad, p, args=(n, S, iu, ju), jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-11})
    return r.x, r.fun


def _joint(q, n, mu, iu, ju):
    e, g, dS = _energy_grad(q[:-1], n, q[-1], iu, ju)
    return q[-1] + mu * e, np.append(mu * g, 1.0 + mu * dS)


def _make_feasible(p, n, S, iu, ju, max_tries=40):
    for _ in range(max_tries):
        p, e = _relax(p, n, S, iu, ju)
        if e < TOL:
            return S, p
        S += 0.05
    return math.inf, None


def _shrink(p, S, n, iu, ju, step=0.02, min_step=1e-4):
    """Shrink container while a zero-energy relaxation still exists."""
    while step > min_step:
        St = S - step
        t = p.copy()
        t[:2 * n] *= St / S
        t, e = _relax(t, n, St, iu, ju)
        if e < TOL:
            p, S = t, St
        else:
            step *= 0.5
    return S, p


# ----------------------------------------------------------------------------
# Search moves
# ----------------------------------------------------------------------------
def _random_start(n, rng, iu, ju):
    S = math.sqrt(n) * 1.15
    p = np.concatenate([rng.uniform(0.5, S - 0.5, 2 * n),
                        rng.uniform(0, math.pi / 2, n)])
    q = np.append(p, S)
    for mu in (5, 30, 200, 2000, 2e4):  # squeeze, container size is a variable
        q = minimize(_joint, q, args=(n, mu, iu, ju), jac=True, method="L-BFGS-B",
                     options={"maxiter": 800, "ftol": 1e-14}).x
    S, p = _make_feasible(q[:-1], n, q[-1], iu, ju)
    if p is None:
        return math.inf, None
    return _shrink(p, S, n, iu, ju)


def _hop(p, S, n, rng, iu, ju):
    """Relocate 1-3 random squares, re-relax in a slightly larger box, shrink."""
    q = p.copy()
    idx = rng.choice(n, rng.integers(1, min(n, 3) + 1), replace=False)
    Sg = S + 0.35
    q[:2 * n] *= Sg / S
    q[idx] = rng.uniform(0.5, Sg - 0.5, len(idx))
    q[n + idx] = rng.uniform(0.5, Sg - 0.5, len(idx))
    q[2 * n + idx] = rng.uniform(0, math.pi / 2, len(idx))
    q[:2 * n] += rng.normal(0, 0.03, 2 * n)
    Sg, q = _make_feasible(q, n, Sg, iu, ju, max_tries=20)
    if q is None:
        return math.inf, None
    return _shrink(q, Sg, n, iu, ju, step=0.02, min_step=2e-3)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
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


def pack_squares(n, time_limit=60.0, seed=0, verbose=False):
    """
    Find a small square container holding n unit squares.

    Parameters
    ----------
    n          : number of unit squares
    time_limit : seconds to search (more time => better results)
    seed       : RNG seed

    Returns
    -------
    side    : container side length (an upper bound on the true optimum)
    squares : list of (x, y, angle_deg); angle in [0, 90)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 1.0, [(0.5, 0.5, 0.0)]

    t0 = time.time()
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    best_S, best_p = math.inf, None
    stall = 0

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    cur_S, cur_p = math.inf, None
    while time.time() - t0 < time_limit:
        if cur_p is None or stall >= 15:        # (re)start
            cur_S, cur_p = _random_start(n, rng, iu, ju)
            stall = 0
            log(f"new random start -> {cur_S:.5f}")
        else:                                    # basin hop
            S2, p2 = _hop(cur_p, cur_S, n, rng, iu, ju)
            if S2 < cur_S - 1e-6:
                cur_S, cur_p, stall = S2, p2, 0
            else:
                stall += 1
        if cur_p is not None and cur_S < best_S - 1e-9:
            best_S, best_p = cur_S, cur_p.copy()
            log(f"best side = {best_S:.5f}")

    # final fine polish
    S, p = _shrink(best_p, best_S, n, iu, ju, step=1e-2, min_step=1e-7)
    side = S + 1e-5  # tiny margin to absorb residual numerical overlap
    p, _ = _relax(p, n, side, iu, ju)
    x, y, t = p[:n], p[n:2 * n], p[2 * n:]
    ang = np.degrees(t) % 90.0
    return float(side), [(float(x[i]), float(y[i]), float(ang[i])) for i in range(n)]


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    side, squares = pack_squares(n, time_limit=tl, verbose=True)
    print(f"\nn = {n}: container side ~= {side:.5f}")
    print(f"max violation: {max_violation(side, squares):.2e}")
    try:
        print("image saved to", plot_packing(side, squares, f"packing_n{n}.png"))
    except ImportError:
        print("(install matplotlib to save an image)")
    for i, (x, y, a) in enumerate(squares, 1):
        print(f"  square {i:2d}: center = ({x:8.4f}, {y:8.4f})  angle = {a:7.3f} deg")
