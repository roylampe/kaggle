"""
strict_gaps.py -- make a square packing strictly valid with the smallest possible changes.

Rules applied by finalize():
  * Two squares with DIFFERENT angles must be separated by a small gap (default 1e-6),
    so that rounding errors can never make them overlap.
  * Two squares with the SAME angle get NO gap:
      - axis-aligned squares (angle 0) end up exactly flush: neighbours are exactly
        1.0 apart, e.g. centres 0.5, 1.5, 2.5 (all coordinates are multiples of 2**-40,
        which are exact in floating point, so touching edges are exactly equal);
      - squares sharing a tilted angle sit flush to within 1e-10.
  * Only centres move, and by the smallest amount that works; angles are only merged
    when they are equal within a tolerance.

Used by image_to_csv.py, square_packing.py and build_packings_csv.py.
Requires numpy (scipy is used for speed when there are many squares, but is optional).
"""
import math
from decimal import Decimal

import numpy as np

VERSION = "3.2"          # 3.x: same-angle squares exactly flush; 3.2: numbers written as exact decimals
Q = 2.0 ** -40            # coordinate grid: multiples of Q are exactly representable
ONE = 1 << 40             # 1.0 expressed in units of Q
HALF_PI = math.pi / 2
ANG_TOL = 1e-9            # angles closer than this (mod 90 degrees) count as "the same"


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def half_extent(t):
    """Half-width of the axis-aligned bounding box of a unit square rotated by t."""
    return 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))


def pair_pen(x, y, t, iu, ju):
    """
    Penetration depth of each square pair (positive = overlap, negative = gap) and the unit
    direction, pointing from square i towards square j, of the least-overlap axis.
    """
    dx, dy = x[ju] - x[iu], y[ju] - y[iu]
    ti, tj = t[iu], t[ju]
    best = np.full(len(iu), np.inf)
    bx, by = np.zeros(len(iu)), np.zeros(len(iu))
    for phi in (ti, ti + HALF_PI, tj, tj + HALF_PI):
        ca, sa = np.cos(phi), np.sin(phi)
        ri = 0.5 * (np.abs(np.cos(phi - ti)) + np.abs(np.sin(phi - ti)))
        rj = 0.5 * (np.abs(np.cos(phi - tj)) + np.abs(np.sin(phi - tj)))
        proj = ca * dx + sa * dy
        ov = ri + rj - np.abs(proj)
        better = ov < best
        sg = np.where(proj >= 0, 1.0, -1.0)
        best = np.where(better, ov, best)
        bx = np.where(better, sg * ca, bx)
        by = np.where(better, sg * sa, by)
    return best, bx, by


def near_pairs(x, y, r=math.sqrt(2) + 0.3):
    """Index pairs (i < j) of squares whose centres are closer than r."""
    n = len(x)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n > 200:
        try:
            from scipy.spatial import cKDTree
            pairs = cKDTree(np.column_stack([x, y])).query_pairs(r, output_type="ndarray")
            if len(pairs):
                return pairs[:, 0].astype(int), pairs[:, 1].astype(int)
            return np.array([], dtype=int), np.array([], dtype=int)
        except ImportError:
            pass
    iu, ju = np.triu_indices(n, 1)
    m = np.hypot(x[iu] - x[ju], y[iu] - y[ju]) < r
    return iu[m], ju[m]


def classify(t, iu, ju, gap, flush_gap):
    """Required gap for each pair, plus masks of same-angle pairs and axis-aligned pairs."""
    d = np.abs(t[iu] - t[ju])
    d = np.minimum(d, HALF_PI - d)
    aligned = d < ANG_TOL
    axis = aligned & (t[iu] == 0) & (t[ju] == 0)
    g = np.full(len(iu), float(gap))
    g[aligned] = flush_gap
    g[axis] = 0.0
    return g, aligned, axis


def merge_angles(t, tol):
    """Snap angles near 0/90 degrees to exactly 0 and merge nearly equal angles (mod 90)."""
    t = np.mod(np.asarray(t, float), HALF_PI)
    t = np.where((t < tol) | (t > HALF_PI - tol), 0.0, t)
    nz = np.flatnonzero(t > 0)
    if len(nz):
        order = nz[np.argsort(t[nz])]
        s = t[order]
        start = 0
        for k in range(1, len(s) + 1):
            if k == len(s) or s[k] - s[k - 1] > tol:
                t[order[start:k]] = s[start:k].mean()
                start = k
    return t


# ----------------------------------------------------------------------------
# the three working stages
# ----------------------------------------------------------------------------
def _push(x, y, t, iu, ju, g, sweeps):
    """Push overlapping / too-close pairs apart until every pair meets its required gap."""
    n = len(x)
    if not len(iu):
        return True
    thr = np.where(g > 0, 0.0, 1e-10)            # flush pairs may overlap by < 1e-10 here
    aim = np.where(g > 0, 1.5 * g, 0.0)          # aim a little beyond the required gap
    for _ in range(sweeps):
        pen, bx, by = pair_pen(x, y, t, iu, ju)
        viol = (pen + g) > thr
        if not viol.any():
            return True
        a, b = iu[viol], ju[viol]
        p = np.maximum((pen[viol] + aim[viol]) * 0.5, 0.0)
        mx, my, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
        np.add.at(mx, b, p * bx[viol]); np.add.at(mx, a, -p * bx[viol])
        np.add.at(my, b, p * by[viol]); np.add.at(my, a, -p * by[viol])
        np.add.at(cnt, a, 1); np.add.at(cnt, b, 1)
        x += mx / np.maximum(cnt, 1)
        y += my / np.maximum(cnt, 1)
    return False


def _shift_to_origin(x, y, t):
    h = half_extent(t)
    x -= (x - h).min()
    y -= (y - h).min()


def _compact(x, y, t, snap_tol):
    """
    Close small gaps (< snap_tol) between squares that share an angle by sliding them flush
    (measured in each group's own rotated frame).  Axis-aligned squares near the left/bottom
    edge of the container are put exactly at 0.5.  Returns True if anything moved.
    """
    moved = False
    _shift_to_origin(x, y, t)
    for ang in np.unique(t):
        idx = np.flatnonzero(t == ang)
        c, s = math.cos(ang), math.sin(ang)
        u = c * x[idx] + s * y[idx]
        v = -s * x[idx] + c * y[idx]
        if ang == 0:
            for arr in (u, v):
                m = (np.abs(arr - 0.5) < snap_tol) & (arr != 0.5)
                if m.any():
                    arr[m] = 0.5
                    moved = True
        for p, q in ((u, v), (v, u)):            # slide along u, then along v
            for j in np.argsort(p, kind="stable"):
                m = (p < p[j] - 0.5) & (np.abs(q - q[j]) < 1 + snap_tol)   # face or corner neighbours
                if not m.any():
                    continue
                target = p[m].max() + 1.0
                d = p[j] - target
                if 0 < d < snap_tol:
                    p[j] = target
                    moved = True
        x[idx] = c * u - s * v
        y[idx] = s * u + c * v
    return moved


def _axis_repair(X, Y, ia, ja):
    """Integer fix-up: axis-aligned squares that overlap by rounding are moved up/right just enough."""
    for _ in range(200000):
        dx, dy = X[ja] - X[ia], Y[ja] - Y[ia]
        adx, ady = np.abs(dx), np.abs(dy)
        v = np.flatnonzero((adx < ONE) & (ady < ONE))
        if not len(v):
            return True
        usex = (ONE - adx[v]) <= (ONE - ady[v])
        vx, vy = v[usex], v[~usex]
        if len(vx):
            later = np.where(dx[vx] >= 0, ja[vx], ia[vx])
            early = np.where(dx[vx] >= 0, ia[vx], ja[vx])
            np.maximum.at(X, later, X[early] + ONE)
        if len(vy):
            later = np.where(dy[vy] >= 0, ja[vy], ia[vy])
            early = np.where(dy[vy] >= 0, ia[vy], ja[vy])
            np.maximum.at(Y, later, Y[early] + ONE)
    return False


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def validate(cfg, gap=1e-6, flush_gap=1e-10):
    """Check a configuration against the rules.  Returns (ok, info)."""
    cfg = np.asarray(cfg, float)
    x, y, t = cfg[:, 0], cfg[:, 1], cfg[:, 2]
    iu, ju = near_pairs(x, y)
    info = {"flush_pairs": 0, "angled_pairs": 0, "min_angled_gap": float("nan"), "ok": True}
    if not len(iu):
        return True, info
    pen, _, _ = pair_pen(x, y, t, iu, ju)
    g, aligned, axis = classify(t, iu, ju, gap, flush_gap)
    tilted = aligned & ~axis
    other = ~aligned
    ok = bool((pen[axis] <= 1e-12).all() and (pen[tilted] <= -0.5 * flush_gap).all()
              and (pen[other] <= -0.5 * gap).all())
    info.update(ok=ok, flush_pairs=int((aligned & (pen > -1e-3)).sum()),
                angled_pairs=int((other & (pen > -1e-3)).sum()))
    if other.any():
        info["min_angled_gap"] = float(-pen[other].max())
    return ok, info


def finalize(cfg, gap=1e-6, flush_gap=1e-10, merge_deg=0.0, snap_tol=0.0, sweeps=8000, rounds=4):
    """
    Make a packing strictly valid.

    cfg        rows (x, y, angle_radians)
    gap        gap required between squares at different angles (unit-square lengths)
    flush_gap  gap between squares sharing a tilted angle (default 1e-10: effectively flush)
    merge_deg  merge angles that differ by less than this many degrees (noisy measurements)
    snap_tol   close gaps smaller than this between squares sharing an angle, making them
               flush (measurement noise); 0 disables it
    Returns (cfg, side, info): cfg has the container's lower-left corner at the origin;
    `side` is the side of the tight bounding square.
    """
    cfg = np.array(cfg, float)
    n = len(cfg)
    if n == 0:
        return cfg, 0.0, {"ok": True}
    x, y = cfg[:, 0].copy(), cfg[:, 1].copy()
    t = merge_angles(cfg[:, 2], max(math.radians(merge_deg), ANG_TOL))
    iu, ju = near_pairs(x, y, math.sqrt(2) + 0.6)
    g, _, _ = classify(t, iu, ju, gap, flush_gap)

    _push(x, y, t, iu, ju, g, sweeps)
    if snap_tol > 0:
        for _ in range(rounds):
            if not _compact(x, y, t, snap_tol):
                break
            _push(x, y, t, iu, ju, g, sweeps)

    # quantise to the exact grid, then repair axis-aligned pairs that rounding made overlap
    for attempt in range(6):
        _shift_to_origin(x, y, t)
        X = np.rint(x / Q).astype(np.int64)
        Y = np.rint(y / Q).astype(np.int64)
        ax = (t[iu] == 0) & (t[ju] == 0)
        _axis_repair(X, Y, iu[ax], ju[ax])
        x, y = X * Q, Y * Q
        ok, info = validate(np.column_stack([x, y, t]), gap, flush_gap)
        if ok:
            break
        _push(x, y, t, iu, ju, g, sweeps)       # rare: tighten again, then re-quantise
    out = np.column_stack([x, y, t])
    if snap_tol > 0:
        out = tidy_gaps(out, tol=snap_tol, gap=gap, flush_gap=flush_gap)
        x, y = out[:, 0], out[:, 1]
        ok, info = validate(out, gap, flush_gap)
    h = half_extent(t)
    side = float(max((x + h).max(), (y + h).max()))
    return out, side, info


def tidy_gaps(cfg, tol=1e-3, gap=1e-6, flush_gap=1e-10, max_passes=30):
    """
    Close small gaps (< tol) between face-adjacent axis-aligned squares by sliding one of
    them flush against the other.  A slide is made only if it
      * creates no overlap and keeps the required gap to squares at different angles, and
      * does not open a gap between the moved square and a neighbour it is already flush with.
    Coordinates stay exact.  Returns a new cfg.
    """
    cfg = np.array(cfg, float)
    x, y, t = cfg[:, 0].copy(), cfg[:, 1].copy(), cfg[:, 2]
    n = len(cfg)
    if n < 2 or tol <= 0:
        return cfg
    axis = t == 0
    iu, ju = near_pairs(x, y)
    nbr = [[] for _ in range(n)]
    for a, b in zip(iu, ju):
        nbr[a].append(b)
        nbr[b].append(a)

    def flush_set(m):
        """Axis neighbours that square m currently touches face-to-face."""
        out = set()
        for p in nbr[m]:
            if axis[p]:
                dx, dy = abs(x[p] - x[m]), abs(y[p] - y[m])
                if (abs(dx - 1) < 1e-12 and dy < 1) or (abs(dy - 1) < 1e-12 and dx < 1):
                    out.add(p)
        return out

    def valid(m):
        """No overlap / too small a gap between square m and any neighbour."""
        if not nbr[m]:
            return True
        ps = np.array(nbr[m])
        mm = np.full(len(ps), m)
        pen, _, _ = pair_pen(x, y, t, mm, ps)
        g, aligned, ax = classify(t, mm, ps, gap, flush_gap)
        need = np.where(ax, 1e-12, -0.5 * g)
        return bool((pen <= need).all())

    changed_any = False
    for _ in range(max_passes):
        changed = False
        for i, j in zip(iu, ju):
            if not (axis[i] and axis[j]):
                continue
            for arr, other, in ((x, y), (y, x)):
                d_main, d_cross = arr[j] - arr[i], other[j] - other[i]
                g = abs(d_main) - 1
                if abs(d_cross) < 1 - 1e-9 and 1e-12 < g < tol:
                    lo, hi = (i, j) if d_main > 0 else (j, i)
                    for mover, target in ((lo, arr[hi] - 1.0), (hi, arr[lo] + 1.0)):
                        before = flush_set(mover)
                        old = arr[mover]
                        arr[mover] = target
                        if valid(mover) and before <= flush_set(mover):
                            changed = changed_any = True
                            break
                        arr[mover] = old
        if not changed:
            break
    return np.column_stack([x, y, t]) if changed_any else cfg


def fmt(v):
    """
    Exact decimal expansion of a float (0.0 -> '0', 1.5 -> '1.5').

    A scorer that reads the text as an exact decimal must see EXACTLY the number the packing
    was built with.  The shortest round-trip text (repr) is a *different* decimal number, and
    two squares that touch exactly would then overlap by ~1e-16 or be impossible to certify.
    Coordinates are multiples of 2**-40, so their exact expansion has at most 40 decimals.
    """
    v = float(v)
    if v == 0.0:
        return "0"
    s = format(Decimal(v), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if len(s) <= 90 else repr(v)


def check_text_rows(rows_text):
    """
    Check axis-aligned squares the way an exact-decimal scorer reads the file: parse the TEXT
    of x, y as exact decimals.  Returns (overlapping_pairs, flush_pairs) among squares whose
    angle text is '0'.  Flush pairs only certify when both numbers are exactly representable.
    """
    pts = []
    for x, y, a in rows_text:
        if Decimal(a) == 0:
            pts.append((Decimal(x), Decimal(y)))
    over = flush = 0
    ONE_D = Decimal(1)
    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            dx, dy = abs(pts[j][0] - xi), abs(pts[j][1] - yi)
            if dx >= 2 or dy >= 2:
                continue
            if dx < ONE_D and dy < ONE_D:
                over += 1
            elif dx == ONE_D or dy == ONE_D:
                flush += 1
    return over, flush


def check_rows(rows, gap_tol=1e-9):
    """
    Independent check of (x, y, angle) rows using explicit corner polygons.
    Returns a dict: worst (largest overlap depth over all pairs; <= 0 means no overlap),
    flush (pairs with the same angle that touch), min_angled_gap (smallest gap between
    squares with different angles), side (bounding square side).
    """
    rows = np.asarray(rows, float)
    n = len(rows)
    P = []
    for x, y, a in rows:
        c, s = math.cos(a), math.sin(a)
        P.append(np.array([(x + c * dx - s * dy, y + s * dx + c * dy)
                           for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]))
    iu, ju = near_pairs(rows[:, 0], rows[:, 1], math.sqrt(2) + 1e-6)
    worst, flush, min_gap = -math.inf, 0, math.inf
    for i, j in zip(iu, ju):
        depth = math.inf
        for poly in (P[i], P[j]):
            for k in range(4):
                e = poly[(k + 1) % 4] - poly[k]
                ax = np.array([e[1], -e[0]]) / math.hypot(*e)
                pa, pb = P[i] @ ax, P[j] @ ax
                depth = min(depth, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
        worst = max(worst, depth)
        d = abs(rows[i, 2] - rows[j, 2]) % HALF_PI
        same = min(d, HALF_PI - d) < ANG_TOL
        if same and depth > -gap_tol:
            flush += 1
        if not same:
            min_gap = min(min_gap, -depth)
    allp = np.vstack(P) if P else np.zeros((1, 2))
    return {"worst": worst, "flush": flush,
            "min_angled_gap": (min_gap if min_gap < math.inf else float("nan")),
            "side": float((allp.max(axis=0) - allp.min(axis=0)).max())}
