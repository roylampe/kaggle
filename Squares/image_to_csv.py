"""
Extract a square packing from an SVG or a raster image (PNG/JPG) and write it as a CSV.

SVG input (preferred -- exact): the corners of every square are read straight from
the vector shapes (<rect>, <polygon>, <path>, with any transform / <g> nesting /
<use>), so centres and angles are exact.  No pixel guessing.  Shapes that are
squares of the common size are the unit squares; the smallest axis-aligned square
that contains them all is the container.  If an SVG holds no readable vector
squares (e.g. text-only or embedded bitmap) it is rasterised (needs cairosvg).

Raster input (PNG/JPG).  Expected picture: a square container drawn with a dark outline, containing unit
squares that are each drawn with a dark outline and a fill colour (any colours),
on a light/white background -- like the pictures produced for this puzzle.

CSV columns:  id, n, square_id, x, y, angle
    id         row identifier (0 .. rows-1, or --start-id ...)
    n          puzzle number (default: number of squares found; override with --n)
    square_id  index of the square within the puzzle (0 .. n-1)
    x, y       centre of the square, in UNIT-SQUARE lengths.  The origin is the
               bottom-left corner of the container, y points UP
               (container = [0, side] x [0, side]).
    angle      rotation in RADIANS, counter-clockwise, in [0, pi/2)

How it works
    1. Outline pixels (dark) and background (white) are removed; what is left are
       the square interiors, which the outlines keep apart.
    2. Each interior is one square: its centroid gives the centre, and the
       direction of its edges gives the angle.
    3. The size of one unit square in pixels is estimated from the spacing of
       neighbouring squares (or given with --unit-px / --side), and all
       coordinates are converted from pixels to unit-square lengths.
    4. Strictness (strict_gaps.py): squares at DIFFERENT angles are separated by a small
       gap (--gap, default 1e-6) so rounding can never make them overlap.  Squares with
       the SAME angle get NO gap: axis-aligned squares end up exactly flush (centres such
       as 0.5, 1.5, 2.5 -- coordinates are exact multiples of 2**-40), and squares sharing
       a tilted angle sit flush to within 1e-10.  Measurement noise (pixel rounding) is
       cleaned up with --snap-tol and --merge-angles.
    5. The written CSV is read back and re-checked with an independent overlap test.

Requires: numpy, opencv-python (cv2), strict_gaps.py (same folder);
          cairosvg only for the SVG raster fallback.

EXAMPLES
    python image_to_csv.py packing.svg                       # SVG, exact; writes packing.csv
    python image_to_csv.py packing.svg --csv all.csv --append --n 83
    python image_to_csv.py packing.svg --raster              # force rasterising the SVG
    python image_to_csv.py packing.svg --container 0,0,600,600   # frame in SVG user units
    python image_to_csv.py packing.png                       # raster image, writes packing.csv
    python image_to_csv.py packing.png --csv out.csv
    python image_to_csv.py packing.png --csv all.csv --append          # add puzzle to a file
    python image_to_csv.py packing.png --n 83                          # set the puzzle number
    python image_to_csv.py packing.png --csv all.csv --append --start-id 5000
    python image_to_csv.py packing.png --side 9.6593         # container side known (units)
    python image_to_csv.py packing.png --unit-px 58.5        # unit square size known (pixels)
    python image_to_csv.py packing.png --debug check.png     # overlay for visual checking
    python image_to_csv.py packing.png --gap 1e-5            # bigger gap between squares at different angles
    python image_to_csv.py packing.png --snap-tol 0.02       # close noise gaps up to 0.02 between same-angle squares
    python image_to_csv.py packing.png --merge-angles 1      # treat angles within 1 degree as equal
    python image_to_csv.py packing.png --raw                 # keep the raw measured numbers
    python image_to_csv.py packing.png --clockwise           # clockwise angles
    python image_to_csv.py packing.png --container 14,16,581,583   # frame (x0,y0,x1,y1) in pixels
    python image_to_csv.py --help
"""
import argparse
import csv
import gzip
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict_gaps import VERSION as STRICT_VERSION      # noqa: E402
from strict_gaps import check_rows, check_text_rows, finalize, fmt      # noqa: E402

VERSION = "3.3"      # 3.x: same-angle squares flush; 3.3: survives a float round trip (text_safe)


# ----------------------------------------------------------------------------
# Image analysis
# ----------------------------------------------------------------------------
def load_rgb(path, svg_width=1200):
    if path.lower().endswith((".svg", ".svgz")):
        try:
            import cairosvg
        except ImportError:
            sys.exit("Rasterising an SVG needs cairosvg:  pip install cairosvg")
        png = cairosvg.svg2png(url=path, output_width=svg_width, background_color="white")
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:  # composite transparent pixels onto white
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def find_frame(dark, override=None):
    """
    Container frame as (x0, y0, x1, y1): centres of the outline strokes, in pixel
    index coordinates.  Found from the bounding box of the dark pixels.
    """
    if override:
        return tuple(float(v) for v in override)
    ys, xs = np.where(dark)
    if len(xs) == 0:
        sys.exit("No dark outline pixels found (try --dark-threshold).")
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    def run_len(line, start, step):          # thickness of the stroke at the edge
        k = 0
        while 0 <= start + step * k < len(line) and line[start + step * k]:
            k += 1
        return k

    midr, midc = (ymin + ymax) // 2, (xmin + xmax) // 2
    tl = run_len(dark[midr], xmin, +1)
    tr = run_len(dark[midr], xmax, -1)
    tt = run_len(dark[:, midc], ymin, +1)
    tb = run_len(dark[:, midc], ymax, -1)
    return (xmin + (tl - 1) / 2.0, ymin + (tt - 1) / 2.0,
            xmax - (tr - 1) / 2.0, ymax - (tb - 1) / 2.0)


def contour_angle(mask):
    """Rotation of a (roughly square) blob, CCW in image-flipped (math) sense, [0, pi/2)."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    P = max(cs, key=len)[:, 0, :].astype(float)
    k = 4
    if len(P) < 2 * k + 3:
        (_, _), (_, _), a = cv2.minAreaRect(P.astype(np.float32))
        return math.radians(-a) % (math.pi / 2)
    d = np.roll(P, -k, axis=0) - np.roll(P, k, axis=0)       # tangent estimate
    theta = np.arctan2(-d[:, 1], d[:, 0])                     # flip y: math orientation
    z = np.exp(4j * theta).mean()                             # 4-fold symmetric mean
    return (math.atan2(z.imag, z.real) / 4.0) % (math.pi / 2)


def extract(rgb, dark_thr=75, white_thr=235, frame=None, erode=1):
    dark = rgb.max(axis=2) < dark_thr
    white = rgb.min(axis=2) > white_thr
    x0, y0, x1, y1 = find_frame(dark, frame)

    fill = ((~dark) & (~white)).astype(np.uint8)
    if erode > 0:
        fill = cv2.erode(fill, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fill, connectivity=4)

    comps = []  # (cx, cy, area, angle_rad)
    inside = lambda cx, cy: x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:                         # specks from anti-aliasing
            continue
        m = (lab == i).astype(np.uint8)
        mom = cv2.moments(m, binaryImage=True)
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        if not inside(cx, cy):
            continue
        comps.append((cx, cy, float(area), contour_angle(m)))
    return comps, (x0, y0, x1, y1)


def estimate_unit(comps, frame, unit_px=None, side=None):
    """Pixels per unit-square length."""
    if unit_px:
        return float(unit_px)
    if side:
        return (frame[2] - frame[0]) / float(side)
    c = np.array([[a, b] for a, b, _, _ in comps])
    if len(c) < 2:
        sys.exit("Need at least two squares to estimate the unit size; use --unit-px or --side.")
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)                        # neighbouring squares touch => distance = 1 unit
    med = np.median(nn)
    return float(np.median(nn[nn < 1.15 * med]))


# ----------------------------------------------------------------------------
# Geometry rules (gaps only between squares at different angles) live in strict_gaps.py
# ----------------------------------------------------------------------------
def read_back_text(path, count):
    """The last `count` rows of the CSV as text (x, y, angle) -- exactly what a scorer parses."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))[-count:]
    return [(r["x"], r["y"], r["angle"]) for r in rows]


def read_back(path, count):
    """The last `count` rows of the CSV as written (x, y, angle)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))[-count:]
    return [(float(r["x"]), float(r["y"]), float(r["angle"])) for r in rows]


# ----------------------------------------------------------------------------
# SVG (vector) reader
# ----------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
_PATH_TOK = re.compile(r"[A-Za-z]|" + _NUM)
_SKIP_TAGS = {"defs", "clippath", "mask", "pattern", "symbol", "marker", "metadata",
              "title", "desc", "style", "script", "lineargradient", "radialgradient", "filter"}


def _nums(text):
    return [float(v) for v in _NUM_RE.findall(text or "")]


def parse_transform(text):
    """SVG transform attribute -> 3x3 matrix (applied right-to-left, as in SVG)."""
    M = np.eye(3)
    for name, args in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", text or ""):
        a, name = _nums(args), name.lower()
        T = np.eye(3)
        if name == "matrix" and len(a) == 6:
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate" and a:
            T[0, 2], T[1, 2] = a[0], (a[1] if len(a) > 1 else 0.0)
        elif name == "scale" and a:
            T[0, 0], T[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        elif name == "rotate" and a:
            t = math.radians(a[0])
            R = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            if len(a) >= 3:
                T1, T2 = np.eye(3), np.eye(3)
                T1[0, 2], T1[1, 2], T2[0, 2], T2[1, 2] = a[1], a[2], -a[1], -a[2]
                R = T1 @ R @ T2
            T = R
        elif name == "skewx" and a:
            T[0, 1] = math.tan(math.radians(a[0]))
        elif name == "skewy" and a:
            T[1, 0] = math.tan(math.radians(a[0]))
        M = M @ T
    return M


def path_polygons(d):
    """Sub-paths of a path 'd' made only of M/L/H/V/Z -> list of point lists (None if curves)."""
    toks = _PATH_TOK.findall(d or "")
    polys, cur = [], []
    x = y = sx = sy = 0.0
    i, cmd = 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur:
                    polys.append(cur)
                    cur = []
                x, y = sx, sy
            elif cmd not in "MmLlHhVv":
                return None                      # curves / arcs: not a polygon
            continue
        if cmd is None:
            return None
        rel, c = cmd.islower(), cmd.upper()
        try:
            if c == "M":
                nx, ny = float(toks[i]), float(toks[i + 1])
                i += 2
                if rel:
                    nx, ny = nx + x, ny + y
                if cur:
                    polys.append(cur)
                cur, x, y, sx, sy = [(nx, ny)], nx, ny, nx, ny
                cmd = "l" if rel else "L"        # extra pairs after M are lineto
                continue
            if c == "L":
                nx, ny = float(toks[i]), float(toks[i + 1])
                i += 2
                x, y = (x + nx, y + ny) if rel else (nx, ny)
            elif c == "H":
                nx = float(toks[i]); i += 1
                x = x + nx if rel else nx
            elif c == "V":
                ny = float(toks[i]); i += 1
                y = y + ny if rel else ny
        except (ValueError, IndexError):
            return None
        if not cur:
            cur = [(sx, sy)]
        cur.append((x, y))
    if cur:
        polys.append(cur)
    return polys


def _local(tag):
    return tag.rsplit("}", 1)[-1].lower()


def _hidden(el):
    return (el.get("display") == "none" or "display:none" in (el.get("style") or "").replace(" ", ""))


def svg_polygons(root):
    """All polygons in the SVG as arrays of points in final (viewer) coordinates."""
    ids = {el.get("id"): el for el in root.iter() if el.get("id")}
    out = []

    def pts_of(el, M):
        tag = _local(el.tag)
        raw = None
        if tag == "rect":
            def val(k):
                v = el.get(k, "0")
                return None if "%" in v else (_nums(v) or [0.0])[0]
            x, y, w, h = val("x"), val("y"), val("width"), val("height")
            if None in (x, y, w, h) or w <= 0 or h <= 0:
                return []
            raw = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
        elif tag in ("polygon", "polyline"):
            v = _nums(el.get("points"))
            raw = [list(zip(v[0::2], v[1::2]))]
        elif tag == "path":
            raw = path_polygons(el.get("d")) or []
        else:
            return []
        res = []
        for poly in raw:
            if len(poly) >= 3:
                P = np.array([[px, py, 1.0] for px, py in poly]) @ M.T
                res.append(P[:, :2])
        return res

    def walk(el, M, depth=0):
        if depth > 50 or _local(el.tag) in _SKIP_TAGS or _hidden(el):
            return
        M = M @ parse_transform(el.get("transform"))
        tag = _local(el.tag)
        if tag == "use":
            href = el.get("{http://www.w3.org/1999/xlink}href") or el.get("href") or ""
            ref = ids.get(href.lstrip("#"))
            if ref is not None:
                T = np.eye(3)
                T[0, 2], T[1, 2] = (_nums(el.get("x")) or [0.0])[0], (_nums(el.get("y")) or [0.0])[0]
                walk_ref(ref, M @ T, depth + 1)
            return
        out.extend(pts_of(el, M))
        for ch in el:
            walk(ch, M, depth + 1)

    def walk_ref(ref, M, depth):                 # a referenced element ignores the _SKIP_TAGS test
        if _local(ref.tag) == "symbol":
            for ch in ref:
                walk(ch, M, depth)
        else:
            walk(ref, M, depth)

    walk(root, np.eye(3))
    return out


def as_square(P):
    """If polygon P (k x 2) is a square return (centre, side, angle_rad in [0, pi/2)), else None."""
    pts = []
    for p in P:
        if not pts or np.hypot(*(p - pts[-1])) > 1e-9:
            pts.append(p)
    if len(pts) > 1 and np.hypot(*(pts[0] - pts[-1])) <= 1e-9:
        pts.pop()
    if len(pts) != 4:
        return None
    Q = np.array(pts)
    e = np.roll(Q, -1, axis=0) - Q
    sides = np.hypot(e[:, 0], e[:, 1])
    if sides.min() < 1e-9 or sides.max() / sides.min() > 1.03:
        return None
    d1, d2 = Q[2] - Q[0], Q[3] - Q[1]
    if abs(np.hypot(*d1) - np.hypot(*d2)) > 0.03 * sides.mean():                 # right angles
        return None
    if np.hypot(*((Q[0] + Q[2]) / 2 - (Q[1] + Q[3]) / 2)) > 0.03 * sides.mean():  # parallelogram
        return None
    z = np.exp(4j * np.arctan2(-e[:, 1], e[:, 0])).sum()   # y flipped -> maths orientation
    return Q.mean(axis=0), float(sides.mean()), (math.atan2(z.imag, z.real) / 4.0) % (math.pi / 2), Q


def from_svg(path, container=None, unit=None, side=None):
    """
    Read the packing from SVG vector shapes.
    Returns (cfg, side_units, info) where cfg rows are (x, y, angle) in unit-square
    lengths (origin bottom-left, y up), or None if no vector squares were found.
    """
    opener = gzip.open if path.lower().endswith(".svgz") else open
    with opener(path, "rb") as f:
        root = ET.fromstring(f.read())
    shapes = [sq for sq in (as_square(P) for P in svg_polygons(root)) if sq is not None]
    if len(shapes) < 1:
        return None

    sides = np.array([s[1] for s in shapes])
    if unit is None:                                 # most common side length = unit square
        counts = [(np.abs(sides / v - 1) < 0.02).sum() for v in sides]
        best = int(np.argmax(counts))                # ties -> first (usually the smallest drawn)
        grp = np.abs(sides / sides[best] - 1) < 0.02
        unit = float(np.median(sides[grp]))
    isunit = np.abs(sides / unit - 1) < 0.03
    notes = []

    # ---- container
    others = [s for s, u in zip(shapes, isunit) if not u]
    units = [s for s, u in zip(shapes, isunit) if u]
    frame = None
    if container:
        frame = tuple(float(v) for v in container)
    else:
        allv = np.vstack([s[3] for s in units])
        lo, hi = allv.min(axis=0), allv.max(axis=0)
        tol = 0.02 * unit
        cands = []
        for c, sd, ang, Q in others:
            axis_aligned = min(ang, math.pi / 2 - ang) < math.radians(0.1)
            qlo, qhi = Q.min(axis=0), Q.max(axis=0)
            if axis_aligned and np.all(qlo <= lo + tol) and np.all(qhi >= hi - tol):
                cands.append((sd, Q))
        if not cands and len(units) == 2:            # n = 1: container is also unit-sized
            a, b = units
            if np.allclose(a[3].min(0), b[3].min(0), atol=tol) and np.allclose(a[3].max(0), b[3].max(0), atol=tol):
                cands.append((a[1], a[3]))      # first shape is the container ...
                units = [b]                     # ... the second is the single unit square
        if cands:
            Q = min(cands, key=lambda c: c[0])[1]
            frame = (Q[:, 0].min(), Q[:, 1].min(), Q[:, 0].max(), Q[:, 1].max())
        else:
            frame = (lo[0], lo[1], hi[0], hi[1])
            notes.append("no container shape found: using the bounding box of the squares "
                         "(pass --container to set it)")
    x0, y0, x1, y1 = frame
    if side:
        unit = (x1 - x0) / side
    if abs((x1 - x0) - (y1 - y0)) > 0.01 * (x1 - x0):
        notes.append(f"container is not square ({x1 - x0:.3f} x {y1 - y0:.3f}); using width")

    # ---- unit squares (dedupe drawn-twice shapes)
    cfg = []
    for c, sd, ang, Q in units:
        row = [(c[0] - x0) / unit, (y1 - c[1]) / unit, ang]
        if not any(np.hypot(row[0] - r[0], row[1] - r[1]) < 0.01 and
                   min(abs(row[2] - r[2]), math.pi / 2 - abs(row[2] - r[2])) < math.radians(0.5)
                   for r in cfg):
            cfg.append(row)
    info = {"unit": unit, "notes": notes, "n_shapes": len(shapes), "frame": frame}
    return np.array(cfg), (x1 - x0) / unit, info


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(cfg, n_puzzle, path, append=False, start_id=None, clockwise=False):
    next_id, header = 0, True
    if append and os.path.exists(path) and os.path.getsize(path) > 0:
        header = False
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            next_id = int(rows[-1][0]) + 1
    if start_id is not None:
        next_id = int(start_id)
    with open(path, "a" if append else "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if header:
            w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for sid, (x, y, t) in enumerate(cfg):
            a = (-t) % (math.pi / 2) if clockwise else t
            if a < 1e-9 or a > math.pi / 2 - 1e-9:
                a = 0.0
            w.writerow([next_id + sid, n_puzzle, sid, fmt(x), fmt(y), fmt(a)])
    return next_id, next_id + len(cfg) - 1


def draw_debug(rgb, cfg, frame, unit, path):
    """Overlay the reconstructed squares (red) and index numbers on the image."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    x0, y0, x1, y1 = frame
    for i, (x, y, t) in enumerate(cfg):
        px, py = x0 + x * unit, y1 - y * unit
        c, s = math.cos(t), math.sin(t)
        pts = np.array([[px + unit * (c * dx - s * dy), py - unit * (s * dx + c * dy)]
                        for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (int(px) - 8, int(py) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# ----------------------------------------------------------------------------
def from_raster(args):
    rgb = load_rgb(args.image, args.raster_width)
    frame_override = [float(v) for v in args.container.split(",")] if args.container else None
    comps, frame = extract(rgb, dark_thr=args.dark_threshold, frame=frame_override)
    if not comps:
        sys.exit("No squares found.")
    unit = estimate_unit(comps, frame, args.unit_px, args.side)
    x0, y0, x1, y1 = frame
    side = (x1 - x0) / unit
    cfg = np.array([[(cx - x0) / unit, (y1 - cy) / unit, ang] for cx, cy, _, ang in comps])
    order = np.lexsort((cfg[:, 0], np.round(cfg[:, 1], 3)))      # bottom-to-top, left-to-right
    cfg = cfg[order]
    areas = np.array([c[2] for c in comps])[order]
    med = np.median(areas)
    odd = [i for i, a in enumerate(areas) if a < 0.85 * med or a > 1.15 * med]
    if odd:
        print(f"WARNING: {len(odd)} region(s) have an unusual size (hidden / merged squares?): "
              f"square_id {odd}", file=sys.stderr)
    return cfg, side, f"raster, {unit:.3f} px per unit square", (rgb, frame, unit)


def main():
    ap = argparse.ArgumentParser(
        description="Extract a square packing from an SVG or image into the packing CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("EXAMPLES")[1] if "EXAMPLES" in __doc__ else "")
    ap.add_argument("--version", action="version",
                    version=f"image_to_csv {VERSION} / strict_gaps {STRICT_VERSION}")
    ap.add_argument("image", help="SVG (preferred) or raster picture of the packing (png/jpg)")
    ap.add_argument("--csv", metavar="FILE", help="output CSV (default: <input name>.csv)")
    ap.add_argument("--n", type=int, default=None,
                    help="puzzle number written in column n (default: number of squares found)")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV")
    ap.add_argument("--start-id", type=int, default=None, metavar="ID",
                    help="id of the first row (default 0, or continue when appending)")
    ap.add_argument("--clockwise", action="store_true", help="write clockwise angles")
    ap.add_argument("--side", type=float, default=None,
                    help="container side in unit-square lengths, if known (sets the scale)")
    ap.add_argument("--unit-px", type=float, default=None,
                    help="size of one unit square in pixels (SVG: in SVG user units), if known")
    ap.add_argument("--container", default=None, metavar="X0,Y0,X1,Y1",
                    help="container frame (pixels, or SVG user units) instead of auto-detect")
    ap.add_argument("--dark-threshold", type=int, default=75,
                    help="raster: pixels whose brightest channel is below this are outlines (75)")
    ap.add_argument("--raster", action="store_true",
                    help="SVG only: rasterise it and use the image method instead of vector shapes")
    ap.add_argument("--raster-width", type=int, default=1200,
                    help="SVG only: width in pixels used when rasterising (default 1200)")
    ap.add_argument("--gap", type=float, default=1e-6,
                    help="gap left between squares with DIFFERENT angles, in unit-square "
                         "lengths (default 1e-6).  Squares with the same angle get no gap.")
    ap.add_argument("--snap-tol", type=float, default=0.01, metavar="T",
                    help="measurement noise: gaps smaller than T between squares sharing an "
                         "angle are closed so they sit exactly flush (default 0.01; 0 = off)")
    ap.add_argument("--merge-angles", type=float, default=None, metavar="DEG",
                    help="treat angles closer than DEG degrees as equal (default 0.5 for "
                         "images, 0 for exact SVG shapes)")
    ap.add_argument("--raw", action="store_true",
                    help="write the measured coordinates as they are (may contain tiny overlaps)")
    ap.add_argument("--repair", action="store_true", help=argparse.SUPPRESS)   # old option, now default
    ap.add_argument("--debug", metavar="PNG", help="save an overlay/preview image to check the result")
    args = ap.parse_args()

    is_svg = args.image.lower().endswith((".svg", ".svgz"))
    result = None
    if is_svg and not args.raster:
        container = [float(v) for v in args.container.split(",")] if args.container else None
        result = from_svg(args.image, container, args.unit_px, args.side)
        if result is None:
            print("No vector squares found in the SVG; falling back to rasterising it.", file=sys.stderr)
    if result is not None:
        cfg, side, info = result
        order = np.lexsort((cfg[:, 0], np.round(cfg[:, 1], 3)))
        cfg = cfg[order]
        for note in info["notes"]:
            print("WARNING:", note, file=sys.stderr)
        method = f"SVG vector shapes (exact), unit square = {info['unit']:.4f} user units"
        S = 640 - 40                                         # blank preview canvas
        canvas = (np.full((640, 640, 3), 255, np.uint8), (20.0, 20.0, 20.0 + S, 20.0 + S), S / side)
    else:
        cfg, side, method, canvas = from_raster(args)

    raw = check_rows(cfg)                                     # signed: negative = gap
    if not args.raw:
        merge = args.merge_angles if args.merge_angles is not None else (0.0 if result is not None else 0.5)
        cfg, side, _ = finalize(cfg, gap=args.gap, merge_deg=merge, snap_tol=args.snap_tol)

    n_puzzle = args.n if args.n is not None else len(cfg)
    if n_puzzle != len(cfg):
        print(f"WARNING: --n {n_puzzle} but {len(cfg)} squares were found.", file=sys.stderr)
    out = args.csv or os.path.splitext(args.image)[0] + ".csv"
    first, last = write_csv(cfg, n_puzzle, out, args.append, args.start_id, args.clockwise)
    if args.debug:
        rgb, frame, unit = canvas
        if result is not None:                               # SVG: draw container on blank canvas
            cv2.rectangle(rgb, (int(frame[0]), int(frame[1])), (int(frame[2]), int(frame[3])), (0, 0, 0), 2)
        draw_debug(rgb, cfg, frame, unit, args.debug)

    print(f"version           : image_to_csv {VERSION} / strict_gaps {STRICT_VERSION}")
    print(f"source            : {method}")
    print(f"squares found     : {len(cfg)}   (n = {n_puzzle})")
    print(f"container side    : {side:.6f} unit squares")
    print(f"angles (degrees)  : " + ", ".join(
        f"{a:g} x{c}" for a, c in zip(*np.unique(np.round(np.degrees(cfg[:, 2]), 1), return_counts=True))))
    chk = check_rows(read_back(out, len(cfg)))
    status = "OK, no overlaps" if chk["worst"] <= 0 else ("OVERLAPS (raw mode)" if args.raw else "OVERLAPS")
    print(f"overlap as measured: {raw['worst']:+.2e}  (positive = overlap, negative = gap)")
    print(f"verified from file : max overlap depth {chk['worst']:+.2e}  -> {status};  "
          f"bounding square side {chk['side']:.6f}")
    over, flush_ok, flush_bad = check_text_rows(read_back_text(out, len(cfg)))
    print(f"scorer-style check : axis-aligned pairs overlapping {over}; exactly flush & certifiable "
          f"{flush_ok}; flush but NOT certifiable {flush_bad}" + ("" if over == 0 and flush_bad == 0 else "   <-- PROBLEM"))
    print(f"contacts           : {chk['flush']} flush (same angle, no gap);  "
          f"smallest gap between different angles: {chk['min_angled_gap']:.2e}")
    print(f"CSV written       : {out}   (ids {first}..{last})")
    if args.debug:
        print(f"debug image       : {args.debug}")


if __name__ == "__main__":
    main()
