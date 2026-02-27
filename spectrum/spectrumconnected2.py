from manimlib import *
import numpy as np

TAU = 2 * np.pi


# ============================================================
# 1) Base spiral (for corridor boundaries)
# ============================================================
def mini_log_spiral_band(
    *,
    turns=3.2,
    n=520,
    r0=0.008,
    growth=0.12,
    band_width=0.06,        # Ancho de la banda espiral
    clockwise=True,
    arm_frac=0.35,
    arm_points=18,
    arm_curvature=0.10,
):
    """
    Crea una banda espiral completa (con outer, center, inner)
    en coordenadas locales.
    Retorna pts_chain = {"c", "o", "i"}
    """
    t = np.linspace(0.0, float(turns) * TAU, int(n))
    sgn = -1.0 if clockwise else 1.0
    th = sgn * t
    r_center = float(r0) * np.exp(float(growth) * t)
    
    # Crear puntos para center, outer e inner
    pts_center = np.stack([r_center * np.cos(th), r_center * np.sin(th), np.zeros_like(r_center)], axis=1)
    
    # Calcular normales para desplazar
    outer_pts = []
    inner_pts = []
    
    for i, p in enumerate(pts_center):
        if i < len(pts_center) - 1:
            t_vec = pts_center[i+1] - pts_center[i]
        else:
            t_vec = pts_center[i] - pts_center[i-1]
        
        norm = np.linalg.norm(t_vec)
        if norm > 1e-6:
            t_vec = t_vec / norm
            n_vec = np.array([-t_vec[1], t_vec[0], 0.0])
        else:
            n_vec = np.array([0.0, 1.0, 0.0])
        
        outer_pts.append(p + n_vec * band_width)
        inner_pts.append(p - n_vec * band_width)
    
    pts_outer = np.array(outer_pts)
    pts_inner = np.array(inner_pts)
    
    # --- Añadir el "brazo" a todas las componentes ---
    p_end_c = pts_center[-1]
    p_end_o = pts_outer[-1]
    p_end_i = pts_inner[-1]
    
    v_end = pts_center[-1] - pts_center[-2]
    v_end /= (np.linalg.norm(v_end) + 1e-9)
    n_end = np.array([-v_end[1], v_end[0], 0.0])
    
    R_end = np.linalg.norm(p_end_c[:2])
    L = float(arm_frac) * (R_end + 1e-9)
    
    us = np.linspace(0.0, 1.0, int(arm_points))
    
    arm_c = []
    arm_o = []
    arm_i = []
    
    for u in us[1:]:
        forward = u * L * v_end
        bulge = float(arm_curvature) * L * (4*u*(1-u)) * n_end
        
        arm_c.append(p_end_c + forward + bulge)
        arm_o.append(p_end_o + forward + bulge)
        arm_i.append(p_end_i + forward + bulge)
    
    if arm_c:
        pts_center = np.vstack([pts_center, np.array(arm_c)])
        pts_outer = np.vstack([pts_outer, np.array(arm_o)])
        pts_inner = np.vstack([pts_inner, np.array(arm_i)])
    
    return {
        "c": pts_center,
        "o": pts_outer,
        "i": pts_inner
    }


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

    pts_c = 0.5 * (pts_out + pts_in)
    v = (pts_out - pts_in)
    width = np.linalg.norm(v, axis=1)
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
# 2) Funciones para colocar bandas completas en el corredor
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


def place_band_on_centerline(
    band_chain,           # Diccionario con {"c", "o", "i"}
    *,
    center_pt,
    tangent_vec,
    scale=1.0,
    mirror=False,
    extra_rot=0.0,
):
    """
    Coloca una banda COMPLETA (outer, center, inner) en el corredor.
    """
    result = {}
    
    for key, pts in band_chain.items():
        pts_copy = pts.copy()
        
        if mirror:
            pts_copy[:, 0] *= -1.0
        
        pts_copy *= float(scale)
        
        ang = np.arctan2(tangent_vec[1], tangent_vec[0]) + float(extra_rot)
        pts_copy = rotate2d(pts_copy, ang)
        pts_copy += center_pt
        
        result[key] = pts_copy
    
    return result


# ============================================================
# 3) Scene: Corredor con mini-espirales que tienen cúspide interna
# ============================================================
class CorridorWithCuspedSpirals(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.10)
        self.add(plane)

        title = Text(
            "Mini-espirales con CÚSPIDE INTERNA (outer/inner)", 
            color=YELLOW
        ).scale(0.55).to_edge(UP)
        self.add(title)

        stroke_b = 3.0
        target_size = 6.7
        tilt = 0.18

        # ---- Corredor (boundaries) ----
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

        # Corredor en blanco
        outer = vmob_from_pts(pts_out, color=WHITE, width=stroke_b, opacity=0.85, smooth=True)
        inner = vmob_from_pts(pts_in,  color=WHITE, width=stroke_b, opacity=0.85, smooth=True)

        # ---- Mini-espiral BASE con banda completa ----
        base_band = mini_log_spiral_band(
            turns=3.2,
            n=320,
            r0=0.008,
            growth=0.12,
            band_width=0.03,        # Ancho de la banda mini-espiral
            clockwise=True,
            arm_frac=0.45,
            arm_points=22,
            arm_curvature=0.12,
        )

        # Radio máximo para escalado
        Rmax0 = np.max(np.linalg.norm(base_band["c"][:, :2], axis=1)) + 1e-9

        # Índices a lo largo del corredor
        step = 280
        start_i = 150
        end_i = len(pts_c) - 150
        idxs = list(range(start_i, end_i, step))

        touch_k = 0.98   # factor de tangencia
        band_k = 0.40     # mantener dentro del corredor

        spirals_group = VGroup()
        
        for k, i in enumerate(idxs):
            c0 = pts_c[i]
            w_local = width[i]

            # Calcular escala basada en vecinos para tangencia
            if k < len(idxs) - 1:
                c1 = pts_c[idxs[k + 1]]
            else:
                c1 = pts_c[idxs[k - 1]]

            d = np.linalg.norm(c1 - c0)

            # Escala para tangencia
            s_touch = (0.5 * d / Rmax0) * touch_k
            # Escala para caber dentro del corredor
            s_band = (band_k * w_local / Rmax0)

            s = min(s_touch, s_band)

            t = unit_tangent(pts_c, i)

            mirror = (k % 2 == 1)  # alternar orientación

            # Colocar la banda COMPLETA en el corredor
            placed_band = place_band_on_centerline(
                base_band,
                center_pt=c0,
                tangent_vec=t,
                scale=s,
                mirror=mirror,
                extra_rot=0.0,
            )

            # Crear mobjects para outer e inner de cada mini-espiral
            colors = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL]
            color = colors[k % len(colors)]
            
            spiral_outer = vmob_from_pts(
                placed_band["o"], 
                color=color, 
                width=2.5, 
                opacity=0.9, 
                smooth=True
            )
            spiral_inner = vmob_from_pts(
                placed_band["i"], 
                color=color, 
                width=2.5, 
                opacity=0.9, 
                smooth=True
            )
            
            spirals_group.add(spiral_outer, spiral_inner)

        # Grupo final con todo
        group = VGroup(outer, inner, spirals_group).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        # Animación
        self.play(ShowCreation(outer), ShowCreation(inner), run_time=1.2)
        
        # Mostrar mini-espirales una por una
        for mob in spirals_group:
            self.play(ShowCreation(mob), run_time=0.4)
            self.wait(0.1)

        # Explicación
        info = Text(
            "CADA mini-espiral tiene su propia estructura (outer/inner)\n"
            "y toca a sus vecinas en un punto",
            font_size=22,
            color=GREEN
        ).to_edge(DOWN)
        self.add(info)

        self.wait(2)


# ============================================================
# 4) Scene con zoom para ver la cúspide interna
# ============================================================
class ZoomedCuspedSpirals(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.10)
        self.add(plane)

        title = Text("Zoom: CÚSPIDE INTERNA visible", color=YELLOW).scale(0.55).to_edge(UP)
        self.add(title)

        stroke_b = 3.0
        target_size = 8.0
        tilt = 0.18

        # ---- Corredor ----
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

        outer = vmob_from_pts(pts_out, color=WHITE, width=stroke_b, opacity=0.5, smooth=True)
        inner = vmob_from_pts(pts_in,  color=WHITE, width=stroke_b, opacity=0.5, smooth=True)

        # ---- Mini-espiral BASE ----
        base_band = mini_log_spiral_band(
            turns=3.5,
            n=400,
            r0=0.008,
            growth=0.12,
            band_width=0.04,
            clockwise=True,
            arm_frac=0.5,
            arm_points=25,
            arm_curvature=0.15,
        )

        Rmax0 = np.max(np.linalg.norm(base_band["c"][:, :2], axis=1)) + 1e-9

        # Colocar solo 3 mini-espirales para ver detalles
        idxs = [250, 550, 850]
        
        spirals_group = VGroup()
        
        for k, i in enumerate(idxs):
            c0 = pts_c[i]
            w_local = width[i]
            t = unit_tangent(pts_c, i)
            
            # Escala para que quepan
            s = (0.4 * w_local / Rmax0)
            
            placed_band = place_band_on_centerline(
                base_band,
                center_pt=c0,
                tangent_vec=t,
                scale=s,
                mirror=(k % 2 == 1),
                extra_rot=0.0,
            )
            
            colors = [BLUE, RED, GREEN]
            
            # Mostrar outer, inner y center para ver la estructura
            spiral_outer = vmob_from_pts(
                placed_band["o"], 
                color=colors[k], 
                width=3.0, 
                opacity=1.0, 
                smooth=True
            )
            spiral_inner = vmob_from_pts(
                placed_band["i"], 
                color=colors[k], 
                width=3.0, 
                opacity=1.0, 
                smooth=True
            )
            spiral_center = vmob_from_pts(
                placed_band["c"], 
                color=YELLOW, 
                width=1.5, 
                opacity=0.8, 
                smooth=True
            )
            
            spirals_group.add(spiral_outer, spiral_inner, spiral_center)

        group = VGroup(outer, inner, spirals_group).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        self.play(ShowCreation(outer), ShowCreation(inner), run_time=1.0)
        self.play(ShowCreation(spirals_group), run_time=2.0)

        # Leyenda
        legend = VGroup(
            Text("Outer de mini-espiral", color=BLUE, font_size=18),
            Text("Inner de mini-espiral", color=RED, font_size=18),
            Text("Center (línea guía)", color=YELLOW, font_size=18),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(DL)
        
        self.add(legend)
        self.wait(2)