from manimlib import *
import numpy as np

TAU = 2 * np.pi


# ============================================================
# 1) Precompute + band builder (AUTOCONTENIDO)
# ============================================================
def _select_by_turn_window(turn_count, t0, t1, min_points=8):
    a = float(min(t0, t1))
    b = float(max(t0, t1))
    idx = np.where((turn_count >= a) & (turn_count <= b))[0]
    if len(idx) < min_points:
        raise ValueError(
            f"Turn window too small/out of range: [{a:.3f},{b:.3f}] -> {len(idx)} pts. "
            f"Increase x_max/n or widen window."
        )
    return idx


def _spiral_precompute(
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
    return x, r, theta, turn_count


def _to_pts(rr, th):
    return np.stack([rr * np.cos(th), rr * np.sin(th), np.zeros_like(rr)], axis=1)


def build_band(
    *,
    x_min=-1.05,
    x_max=6.10,
    n=26000,
    base_turn_rate=0.95,
    r_cut=0.0018,
    band_w=0.24,
    center_turn_window=(0.15, 4.60),
    outer_turn_window=(0.40, 4.20),
    inner_turn_window=(0.20, 4.70),
    reverse=True,
):
    """
    Returns (pts_chain, meta) where pts_chain = {"c","o","i"} each Nx3.
    Independent turn windows for c/o/i (your current architecture).
    """
    x, r, theta, turn_count = _spiral_precompute(
        x_min=x_min, x_max=x_max, n=n,
        base_turn_rate=base_turn_rate, r_cut=r_cut
    )

    w = float(np.clip(band_w, 0.02, 0.49))
    r_o = r * (1.0 + w)
    r_i = r * (1.0 - w)

    idx_c = _select_by_turn_window(turn_count, center_turn_window[0], center_turn_window[1])
    idx_o = _select_by_turn_window(turn_count, outer_turn_window[0], outer_turn_window[1])
    idx_i = _select_by_turn_window(turn_count, inner_turn_window[0], inner_turn_window[1])

    pts_c = _to_pts(r[idx_c],   theta[idx_c])
    pts_o = _to_pts(r_o[idx_o], theta[idx_o])
    pts_i = _to_pts(r_i[idx_i], theta[idx_i])

    if reverse:
        pts_c = pts_c[::-1]
        pts_o = pts_o[::-1]
        pts_i = pts_i[::-1]

    meta = dict(
        x=x, r=r, theta=theta, turn_count=turn_count, band_w=w,
        idx_c=idx_c, idx_o=idx_o, idx_i=idx_i,
        center_turn_window=center_turn_window,
        outer_turn_window=outer_turn_window,
        inner_turn_window=inner_turn_window,
    )
    return {"c": pts_c, "o": pts_o, "i": pts_i}, meta


def make_band_mobjects(pts_chain, stroke_band=4.0, color=WHITE):
    outer = VMobject().set_points_smoothly(pts_chain["o"])
    inner = VMobject().set_points_smoothly(pts_chain["i"])
    outer.set_stroke(color, width=stroke_band, opacity=0.95)
    inner.set_stroke(color, width=stroke_band, opacity=0.95)
    return outer, inner


# ============================================================
# 2) Funciones para transformación que preserva espiral
# ============================================================
def _tangent_vec(pts, at_start=True):
    if len(pts) < 2:
        return RIGHT
    v = (pts[1] - pts[0]) if at_start else (pts[-1] - pts[-2])
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _normal_vec(pts, at_start=True):
    """Calcula vector normal (rotación 90° CCW de la tangente)."""
    t = _tangent_vec(pts, at_start)
    return np.array([-t[1], t[0], 0.0])


def _rotate_pts(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return pts @ R.T


def _align_vectors(v_from, v_to):
    """Calcula el ángulo de rotación para alinear v_from con v_to."""
    angle_from = np.arctan2(v_from[1], v_from[0])
    angle_to = np.arctan2(v_to[1], v_to[0])
    return angle_to - angle_from


def glue_two_bands_two_spirals(
    A_chain, B_chain,
    glue_key="o",          # "o" (outer) or "i" (inner)
    use_end_of_A=True,     # glue at A end (True) or A start (False)
    use_start_of_B=False,  # ¡IMPORTANTE: False para pegar el FINAL de B!
    flip_B=False,          # ¡IMPORTANTE: False para no invertir!
    angle_bias=0.0,
):
    """
    Transforma B para que sea UNA SOLA continuación de A.
    Para tener SOLO DOS espirales, B debe comenzar desde el punto de unión
    y continuar hacia afuera (como una extensión natural).
    
    Parámetros clave:
    - use_start_of_B=False: pega el FINAL de B (no el principio)
    - flip_B=False: no invierte la dirección
    """
    # NO invertimos B - queremos que continúe en la misma dirección
    if flip_B:
        B_chain = {k: v[::-1].copy() for k, v in B_chain.items()}

    A_pts = A_chain[glue_key]
    B_pts = B_chain[glue_key]

    # Puntos de anclaje
    A_anchor = A_pts[-1] if use_end_of_A else A_pts[0]
    # Usamos el FINAL de B (no el principio) para que continúe desde A
    B_anchor = B_pts[-1] if use_start_of_B else B_pts[0]  # Cambiado: ahora usamos el principio de B
    
    # Tangentes
    A_tan = _tangent_vec(A_pts, at_start=not use_end_of_A)
    B_tan = _tangent_vec(B_pts, at_start=use_start_of_B)
    
    # Para que B continúe en la misma dirección que A, necesitamos que
    # la tangente de B en el punto de unión apunte en la MISMA dirección que A_tan
    # (no opuesta)
    rot_angle = _align_vectors(B_tan, A_tan) + float(angle_bias)
    
    # Aplicar transformación
    out = {}
    for key, pts in B_chain.items():
        # Rotar
        pts_rot = _rotate_pts(pts, rot_angle)
        
        # El punto que debe coincidir con A_anchor es el PRINCIPIO de B
        # (así B se extiende desde A hacia afuera)
        B_anchor_rot = pts_rot[0]  # Siempre usamos el principio después de rotar
        shift = A_anchor - B_anchor_rot
        
        # Trasladar
        out[key] = pts_rot + shift
    
    return out


# ============================================================
# 3) Scene con DOS espirales
# ============================================================
class TwoSpiralBands_Glued_Corrected(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.12)
        self.add(plane)

        title = Text("DOS espirales - Continuación correcta", color=YELLOW).scale(0.58).to_edge(UP)
        self.add(title)

        stroke_band = 4.0
        target_size = 6.3
        tilt = 0.20

        # --- Build A (banda base) ---
        A_pts, A_meta = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            center_turn_window=(0.15, 4.60),
            outer_turn_window=(0.40, 4.20),
            inner_turn_window=(0.20, 4.70),
            reverse=True,
        )

        # --- Build B (banda a pegar) con ventanas que se superponen ligeramente ---
        B_pts, B_meta = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            center_turn_window=(0.10, 4.50),  # Ligeramente diferente
            outer_turn_window=(0.25, 4.00),   # Para que sea una continuación natural
            inner_turn_window=(0.05, 4.50),
            reverse=True,
        )

        # --- Aplicar glue para DOS espirales ---
        B_glued = glue_two_bands_two_spirals(
            A_chain=A_pts,
            B_chain=B_pts,
            glue_key="o",
            use_end_of_A=True,      # Pegar al final de A
            use_start_of_B=False,    # ¡Clave! Pegar el principio de B al final de A
            flip_B=False,            # ¡Clave! No invertir B
            angle_bias=0.0,
        )

        # Crear mobjects
        A_outer, A_inner = make_band_mobjects(A_pts, stroke_band=stroke_band, color=BLUE)
        B_outer, B_inner = make_band_mobjects(B_glued, stroke_band=stroke_band, color=RED)

        # Agrupar y escalar
        group = VGroup(A_outer, A_inner, B_outer, B_inner).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        # Mostrar punto de unión
        # Calcular punto de unión después de todas las transformaciones
        glue_point = group[0].get_end()  # El final de A_outer después de transformaciones
        self.add(Dot(glue_point, radius=0.08, color=YELLOW))
        
        # Etiquetas
        label_A = Text("A", color=BLUE, font_size=36).next_to(group[0].get_center(), UP, buff=0.5)
        label_B = Text("B", color=RED, font_size=36).next_to(group[2].get_center(), DOWN, buff=0.5)
        self.add(label_A, label_B)

        # Animación
        self.play(ShowCreation(A_outer), ShowCreation(A_inner), run_time=1.0)
        self.play(ShowCreation(B_outer), ShowCreation(B_inner), run_time=1.0)
        
        # Explicación
        explanation = Text(
            "SOLO dos espirales: B continúa desde A\n"
            "(use_start_of_B=False, flip_B=False)",
            font_size=24, color=GREEN
        ).to_edge(DOWN)
        self.add(explanation)
        
        self.wait(2)


# ============================================================
# 4) Scene comparativa (opcional)
# ============================================================
class TwoSpiralBands_Comparison(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.12)
        self.add(plane)

        title = Text("Comparación: 2 vs 3 espirales", color=WHITE).scale(0.58).to_edge(UP)
        self.add(title)

        stroke_band = 3.0
        target_size = 5.0

        # --- Build A (base) ---
        A_pts, _ = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            reverse=True,
        )

        # --- Build B ---
        B_pts, _ = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            outer_turn_window=(0.30, 4.10),
            inner_turn_window=(0.10, 4.60),
            reverse=True,
        )

        # --- CASO 1: Original (3 espirales) ---
        B_bad = glue_two_bands_two_spirals(  # Usamos la misma función pero con parámetros diferentes
            A_chain=A_pts,
            B_chain=B_pts,
            glue_key="o",
            use_end_of_A=True,
            use_start_of_B=True,   # Esto causa 3 espirales
            flip_B=True,            # Esto también contribuye
        )

        A_outer1, A_inner1 = make_band_mobjects(A_pts, stroke_band=stroke_band, color=BLUE)
        B_outer1, B_inner1 = make_band_mobjects(B_bad, stroke_band=stroke_band, color=RED)
        
        group1 = VGroup(A_outer1, A_inner1, B_outer1, B_inner1).move_to(ORIGIN)
        size1 = max(group1.get_width(), group1.get_height(), 1e-3)
        group1.scale(target_size / size1 * 0.8).move_to(LEFT * 3.5)
        
        label1 = Text("Original: 3 espirales", color=RED, font_size=24).next_to(group1, DOWN)
        
        self.add(group1, label1)

        # --- CASO 2: Corregido (2 espirales) ---
        B_good = glue_two_bands_two_spirals(
            A_chain=A_pts,
            B_chain=B_pts,
            glue_key="o",
            use_end_of_A=True,
            use_start_of_B=False,   # ¡Clave!
            flip_B=False,            # ¡Clave!
        )

        A_outer2, A_inner2 = make_band_mobjects(A_pts, stroke_band=stroke_band, color=BLUE)
        B_outer2, B_inner2 = make_band_mobjects(B_good, stroke_band=stroke_band, color=RED)
        
        group2 = VGroup(A_outer2, A_inner2, B_outer2, B_inner2).move_to(ORIGIN)
        size2 = max(group2.get_width(), group2.get_height(), 1e-3)
        group2.scale(target_size / size2 * 0.8).move_to(RIGHT * 3.5)
        
        label2 = Text("Corregido: 2 espirales", color=GREEN, font_size=24).next_to(group2, DOWN)
        
        # Punto de unión en caso corregido
        glue_point = group2[0].get_end()
        dot = Dot(glue_point, radius=0.06, color=YELLOW)
        group2.add(dot)
        
        self.add(group2, label2)
        
        # Explicación
        explanation = Text(
            "use_start_of_B=False, flip_B=False  →  B continúa naturalmente desde A",
            font_size=20, color=WHITE
        ).to_edge(DOWN)
        self.add(explanation)
        
        self.wait(3)