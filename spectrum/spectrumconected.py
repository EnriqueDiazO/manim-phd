from manimlib import *
import numpy as np

TAU = 2 * np.pi


# ============================================================
# 1) Base spiral (for corridor boundaries)
# ============================================================
def mini_log_spiral_points(
    *,
    turns=3.2,
    n=520,
    r0=0.008,
    growth=0.12,
    clockwise=True,
    arm_frac=0.35,     # <- largo del brazo relativo al radio final de la espiral
    arm_points=18,     # <- suavidad del brazo
    arm_curvature=0.10 # <- 0 = recto, >0 curva ligera (tipo Fig.5)
):
    """
    Log-spiral from center outward + an "arm" that exits the spiral,
    like the ornament in Fig.5.
    """
    t = np.linspace(0.0, float(turns) * TAU, int(n))
    sgn = -1.0 if clockwise else 1.0
    th = sgn * t
    rr = float(r0) * np.exp(float(growth) * t)

    pts = np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros_like(rr)], axis=1)

    # --- build an arm from the last point, following the final tangent ---
    p_end = pts[-1]
    v_end = pts[-1] - pts[-2]
    v_end /= (np.linalg.norm(v_end) + 1e-9)

    # arm length proportional to final radius
    R_end = np.linalg.norm(p_end[:2])
    L = float(arm_frac) * (R_end + 1e-9)

    # a slightly curved arm: go forward, with a small normal offset
    n_end = np.array([-v_end[1], v_end[0], 0.0])

    us = np.linspace(0.0, 1.0, int(arm_points))
    arm = []
    for u in us[1:]:
        forward = u * L * v_end
        # small bulge: 0 at start/end, max in middle
        bulge = float(arm_curvature) * L * (4*u*(1-u)) * n_end
        arm.append(p_end + forward + bulge)

    if arm:
        pts = np.vstack([pts, np.array(arm)])

    return pts



def spiral_precompute(
    *,
    x_min=-1.05,
    x_max=6.10,
    n=26000,
    base_turn_rate=0.95,
    r_cut=0.0018,
):
    x = np.linspace(x_min, x_max, n)
    dx = (x_max - x_min) / max(n - 1, 1)

    r = np.exp(-np.exp(x))

    if r_cut is not None:
        mask = (r >= float(r_cut))
        x, r = x[mask], r[mask]
        dx = (x[-1] - x[0]) / max(len(x) - 1, 1)

    omega = TAU * float(base_turn_rate) * np.ones_like(x)
    theta = np.cumsum(omega) * dx
    theta -= theta[0]
    turn_count = theta / TAU

    return dict(x=x, dx=dx, r=r, theta=theta, turn_count=turn_count)


def to_pts(rr, th):
    return np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros_like(rr)], axis=1)


def select_by_turn_window(turn_count, t0, t1, min_points=256):
    a = float(min(t0, t1))
    b = float(max(t0, t1))
    idx = np.where((turn_count >= a) & (turn_count <= b))[0]
    if len(idx) < min_points:
        raise ValueError(f"Turn window too small: [{a:.3f},{b:.3f}] -> {len(idx)} pts.")
    return idx


def build_corridor_boundaries(
    pre,
    *,
    band_w=0.24,
    turn_window=(0.20, 4.70),
    reverse=True,
):
    r = pre["r"]
    th = pre["theta"]
    tc = pre["turn_count"]

    idx = select_by_turn_window(tc, turn_window[0], turn_window[1], min_points=512)

    w = float(np.clip(band_w, 0.02, 0.49))
    r_out = r[idx] * (1.0 + w)
    r_in  = r[idx] * (1.0 - w)

    pts_out = to_pts(r_out, th[idx])
    pts_in  = to_pts(r_in,  th[idx])

    if reverse:
        pts_out = pts_out[::-1].copy()
        pts_in  = pts_in[::-1].copy()

    pts_c = 0.5 * (pts_out + pts_in)      # centerline
    v = (pts_out - pts_in)                # width vector (outer-inner)
    width = np.linalg.norm(v, axis=1)     # local band thickness
    return pts_out, pts_in, pts_c, v, width


def vmob_from_pts(pts, *, color=WHITE, width=3.0, opacity=0.95, smooth=True):
    m = VMobject()
    if smooth:
        m.set_points_smoothly(pts)
    else:
        m.set_points_as_corners(pts)
    m.set_stroke(color, width=width, opacity=opacity)
    return m


# ============================================================
# 2) Mini-spiral glyph (local coords) + placement on corridor
# ============================================================


def unit_tangent(pts, i):
    i0 = max(0, i - 1)
    i1 = min(len(pts) - 1, i + 1)
    v = pts[i1] - pts[i0]
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def rotate2d(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return pts @ R.T


def place_glyph_on_centerline(
    pts_glyph,
    *,
    center_pt,
    tangent_vec,
    scale=1.0,
    mirror=False,
    extra_rot=0.0,
):
    pts = pts_glyph.copy()
    if mirror:
        pts[:, 0] *= -1.0

    pts *= float(scale)

    ang = np.arctan2(tangent_vec[1], tangent_vec[0]) + float(extra_rot)
    pts = rotate2d(pts, ang)
    pts += center_pt
    return pts


# ============================================================
# 3) Scene: corridor + fig-5-like spiral ornaments (touching neighbors)
# ============================================================
class Corridor_Fig5LikeSpirals(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.10)
        self.add(plane)

        title = Text("Corridor con mini-espirales (estilo Fig. 5)", color=WHITE).scale(0.55).to_edge(UP)
        self.add(title)

        stroke_b = 3.0
        target_size = 6.7
        tilt = 0.18

        # ---- corridor boundaries ----
        pre = spiral_precompute(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018
        )

        pts_out, pts_in, pts_c, v_width, width = build_corridor_boundaries(
            pre,
            band_w=0.24,
            turn_window=(0.20, 4.70),
            reverse=True,
        )

        outer = vmob_from_pts(pts_out, color=WHITE, width=stroke_b, opacity=0.85, smooth=True)
        inner = vmob_from_pts(pts_in,  color=WHITE, width=stroke_b, opacity=0.85, smooth=True)

        # ---- mini-spiral glyph (more turns) ----
        glyph_base = mini_log_spiral_points(
    turns=3.2,
    n=320,
    r0=0.008,
    growth=0.12,
    clockwise=True,
    arm_frac=0.45,
    arm_points=22,
    arm_curvature=0.12,
)


        # indices along centerline
        step = 320
        start_i = 120
        end_i = len(pts_c) - 120
        idxs = list(range(start_i, end_i, step))

        # radius max of base glyph (local coords) for tangency scaling
        Rmax0 = np.max(np.linalg.norm(glyph_base[:, :2], axis=1)) + 1e-9

        touch_k = 0.99  # tangency safety (<1 avoids overlaps)
        band_k  = 0.46   # keep inside corridor (<0.5 avoids touching boundaries)

        spirals_group = VGroup()
        for k, i in enumerate(idxs):
            c0 = pts_c[i]

            # neighbor center for spacing-based tangency
            if k < len(idxs) - 1:
                c1 = pts_c[idxs[k + 1]]
            else:
                c1 = pts_c[idxs[k - 1]]

            d = np.linalg.norm(c1 - c0)

            # scale needed for neighbors to touch at one point (2*R = d)
            s_touch = (0.5 * d / Rmax0) * touch_k

            # cap scale so glyph stays inside corridor
            s_band = (band_k * width[i] / Rmax0)

            s = min(s_touch, s_band)

            t = unit_tangent(pts_c, i)

            mirror = (k % 2 == 1)

            pts_g = place_glyph_on_centerline(
                glyph_base,
                center_pt=c0,
                tangent_vec=t,
                scale=s,
                mirror=mirror,
                extra_rot=0.0,
            )

            gmob = vmob_from_pts(pts_g, color=WHITE, width=2.6, opacity=0.95, smooth=True)
            spirals_group.add(gmob)

        group = VGroup(outer, inner, spirals_group).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        self.play(ShowCreation(outer), ShowCreation(inner), run_time=1.2)
        self.play(LaggedStart(*[ShowCreation(m) for m in spirals_group], lag_ratio=0.06), run_time=2.2)

        info = Text(
            "Mini-espirales: más vueltas + escala por tangencia (tocan en un punto).",
            font_size=22
        ).to_edge(DOWN)
        self.add(info)

        self.wait(2)
