# paper_spectrum_scenes.py
# ManimGL (manimlib) scenes for "Cauchy's Singular Integral Operator and Its Beautiful Spectrum"
# Run example:
#   manimgl paper_spectrum_scenes.py Fig01_CornerCusp
#   manimgl paper_spectrum_scenes.py Fig02_LogSpiralsDelta

from manimlib import *
import numpy as np

# -------------------------
# Helpers
# -------------------------

def complex_to_point(z: complex) -> np.ndarray:
    return np.array([z.real, z.imag, 0.0])

def log_spiral_points(delta, phi_max=10*np.pi, n=2000, a=0.03):
    phi = np.linspace(0, phi_max, n)
    r = a * np.exp(delta * phi)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, np.zeros_like(x)], axis=1)

def circle_arc_between_pm1(c, n=600, upper=True):
    # Circle centered at (0, c) passing through (-1,0) and (1,0)
    R = np.sqrt(1 + c*c)
    th1 = np.arctan2(-c, 1.0)   # angle to (1,0)
    th2 = np.arctan2(-c, -1.0)  # angle to (-1,0)
    if upper:
        if th2 < th1:
            th2 += 2*np.pi
        th = np.linspace(th1, th2, n)
    else:
        if th1 < th2:
            th1 += 2*np.pi
        th = np.linspace(th1, th2, n)
    x = R*np.cos(th)
    y = c + R*np.sin(th)
    return np.stack([x, y, np.zeros_like(x)], axis=1)

def mobius(zeta):
    return (zeta + 1) / (zeta - 1)

def exp_map(z):
    return np.exp(2*np.pi*z)

def map_indicator_to_leaf(points_z: np.ndarray) -> np.ndarray:
    # points_z: Nx2 or Nx3; interpret as complex z = x + i y
    x = points_z[:, 0]
    y = points_z[:, 1]
    z = x + 1j*y
    zeta = exp_map(z)
    w = mobius(zeta)
    pts = np.stack([w.real, w.imag, np.zeros_like(w.real)], axis=1)
    # remove wild infinities near zeta=1
    good = np.isfinite(pts).all(axis=1) & (np.abs(w - 1) > 1e-6)
    return pts[good]

def point_cloud(points, radius=0.02):
    # render as a bunch of dots
    dots = VGroup(*[Dot(p, radius=radius) for p in points])
    return dots

# -------------------------
# Fig. 1 (corner & cusp)
# -------------------------
class Fig01_CornerCusp(Scene):
    def construct(self):
        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.35)
        self.add(plane)

        # -------------------------
        # (0) Definición previa: punto singular (para curvas parametrizadas)
        # -------------------------
        # Un punto P=γ(t0) es "singular" (en el sentido de parametrización) si γ'(t0)=0.
        singular_def = VGroup(
            Text("Definición (curva parametrizada)").scale(0.42),
            Tex(r"\gamma:I\to\mathbb{R}^2,\quad P=\gamma(t_0).").scale(0.55),
            Tex(r"\text{P es singular si }\gamma'(t_0)=0\ \ (\text{la parametrización no es regular}).").scale(0.55),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UL)

        self.play(FadeIn(singular_def), run_time=0.8)
        self.wait(8.0)
        self.play(FadeOut(singular_def), run_time=0.6)

        # -------------------------
        # Corner: polyline
        # -------------------------
        pL = np.array([-4, 0, 0])
        pV = np.array([0, 1, 0])      # vertex
        pR = np.array([4, -0.5, 0])

        corner = VMobject()
        corner.set_points_as_corners([pL, pV, pR])
        corner.set_stroke(width=6)

        corner_dot = Dot(pV, radius=0.06)
        corner_label = Text("esquina (corner)").scale(0.65).next_to(corner_dot, UP)

        seg_len = 1.2

        v_in = (pV - pL)
        v_in = v_in / np.linalg.norm(v_in)
        v_out = (pR - pV)
        v_out = v_out / np.linalg.norm(v_out)

        # Finite tangent segments at the corner (two one-sided directions)
        corner_tan1 = Line(pV - 0.5*seg_len*v_in, pV + 0.5*seg_len*v_in).set_stroke(width=6)
        corner_tan2 = Line(pV - 0.5*seg_len*v_out, pV + 0.5*seg_len*v_out).set_stroke(width=6)
        corner_tans = VGroup(corner_tan1, corner_tan2).set_color(YELLOW)

        # Runner dots for tangents
        run1 = Dot(corner_tan1.get_start(), radius=0.05).set_color(YELLOW)
        run2 = Dot(corner_tan2.get_start(), radius=0.05).set_color(YELLOW)

        # -------------------------
        # Cusp: parametric curve (t^2, t^3)
        # -------------------------
        def cusp_func(t):
            x = 1.2*(t**2) - 1.5
            y = 0.6*(t**3)
            return np.array([x, y, 0])

        cusp = ParametricCurve(cusp_func, t_range=[-2, 2, 0.01]).set_stroke(width=6)
        cusp_p0 = cusp_func(0)
        cusp_dot = Dot(cusp_p0, radius=0.06)
        cusp_label = Text("cúspide (cusp)").scale(0.65).next_to(cusp_dot, DOWN)

        # Finite tangent at the cusp (horizontal direction)
        cusp_tan = Line(
            cusp_p0 + LEFT*(seg_len/2),
            cusp_p0 + RIGHT*(seg_len/2),
        ).set_stroke(width=6).set_color(YELLOW)

        run3 = Dot(cusp_tan.get_start(), radius=0.05).set_color(YELLOW)

        # -------------------------
        # Texto en español + definiciones simbólicas (autocontenido)
        # -------------------------
        # Nota: aquí γ denota una parametrización local de la curva (por tramos si es necesario).
        txt_corner = VGroup(
            Tex(r"\text{Sea }\gamma\text{ una parametrización local de la curva y }P=\gamma(0).").scale(0.50),
            Text("Esquina: NO hay tangente única en P (hay dos direcciones laterales).").scale(0.40),
            Tex(
                r"\exists\,\tau_- \neq \tau_+:\ "
                r"\lim_{s\to 0^-}\frac{\gamma(s)-\gamma(0)}{\left\lVert \gamma(s)-\gamma(0)\right\rVert}=\tau_-,\ "
                r"\lim_{s\to 0^+}\frac{\gamma(s)-\gamma(0)}{\left\lVert \gamma(s)-\gamma(0)\right\rVert}=\tau_+"
            ).scale(0.50),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UL)

        txt_cusp = VGroup(
            Tex(r"\text{Sea }\gamma\text{ una parametrización local de la curva y }P=\gamma(0).").scale(0.50),
            Text("Cúspide: existe una dirección tangente en P, pero P es singular.").scale(0.40),
            Tex(
                r"\exists\,\tau:\ "
                r"\lim_{t\to 0}\frac{\gamma(t)-\gamma(0)}{\left\lVert \gamma(t)-\gamma(0)\right\rVert}=\tau,"
                r"\qquad \text{pero }\gamma'(0)=0."
            ).scale(0.50),
            Tex(r"\text{Ejemplo: }\gamma(t)=(t^2,t^3)\ \Rightarrow\ \tau=(1,0).").scale(0.50),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UL)

        # -------------------------
        # Animation
        # -------------------------
        self.play(ShowCreation(corner), FadeIn(corner_dot), FadeIn(corner_label), FadeIn(txt_corner))
        self.play(ShowCreation(corner_tans), run_time=0.5)

        # Run along both corner tangents
        self.add(run1, run2)
        self.play(
            run1.animate.move_to(corner_tan1.get_end()),
            run2.animate.move_to(corner_tan2.get_end()),
            run_time=0.8,
            rate_func=linear,
        )
        self.wait(0.2)

        # Morph corner -> cusp; tangents merge into one tangent; switch text
        self.play(
            Transform(corner, cusp),
            Transform(corner_dot, cusp_dot),
            Transform(corner_label, cusp_label),
            Transform(corner_tans, cusp_tan),
            Transform(txt_corner, txt_cusp),
            run_time=1.6,
            rate_func=smooth,
        )

        # Switch runners: keep one runner and traverse the cusp tangent
        self.remove(run2)
        self.play(Transform(run1, run3), run_time=0.3)
        self.play(
            run1.animate.move_to(cusp_tan.get_end()),
            run_time=0.8,
            rate_func=linear,
        )

        self.wait(1)

#################

#######################3

class Fig02_Delta_Teoria_a_Graficas(Scene):
    def construct(self):
        # --- Fondo
        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.20)
        self.add(plane)

        title = Text("Figura 2: espirales logarítmicas (δ) en arcos Carleson").scale(0.58).to_edge(UP)
        self.play(FadeIn(title, shift=0.15*UP), run_time=0.5)

        # ============================================================
        # (A) Fase teórica
        # ============================================================
        expl = VGroup(
            Text("En el artículo se considera el arco (con extremos 0 y 1):").scale(0.42),
            Tex(r"\Gamma=\{0,1\}\ \cup\ \{\, r e^{i\theta(r)}:\ 0<r<1\,\}.").scale(0.60),

            Text("Para animar, usamos una parametrización del arco:").scale(0.42),
            Tex(r"\gamma:(0,1)\to\mathbb{C},\qquad \gamma(r)= r e^{i\theta(r)}.").scale(0.60),

            Text("La fase se define por:").scale(0.42),
            Tex(r"\theta(r)=h(\log|\log r|)\,|\log r|,\qquad h\in C^{1}(\mathbb{R}).").scale(0.58),

            Text("Casos importantes mencionados en el paper:").scale(0.42),
            Tex(
                r"h\equiv 0\Rightarrow \Gamma=[0,1],\qquad "
                r"h\equiv \delta\neq 0\Rightarrow \Gamma\ \text{es una espiral logarítmica.}"
            ).scale(0.54),

            Text("Interpretación: el signo y magnitud de δ controlan si la espiral entra/sale y qué tan rápido.").scale(0.40),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_corner(UL).shift(DOWN*0.35)

        self.play(FadeOut(title, shift=0.15*UP), run_time=0.1)

        self.play(FadeIn(expl), run_time=0.7)
        self.wait(1.5)

        # --- Preparar “mini-resumen” para dejarlo como referencia en una esquina
        resumen = VGroup(
            Text("Modelo (Fig. 2):").scale(0.36),
            Tex(r"\gamma(r)=re^{i\theta(r)},\ \ \theta(r)=h(\log|\log r|)\,|\log r|").scale(0.44),
            Tex(r"h\equiv\delta\neq 0\Rightarrow\ \text{espiral logarítmica}").scale(0.44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10).to_corner(UL)

        # Transición: el bloque grande se transforma en el mini-resumen y se sube a esquina
        self.play(Transform(expl, resumen), run_time=0.9)
        self.wait(0.4)

        # ============================================================
        # (B) Fase gráfica (panel limpio)
        # ============================================================

        # Opcional: “limpiar” un poco el centro (sin borrar el resumen)
        # (Si prefieres, comenta estas dos líneas)
        # self.play(plane.animate.set_stroke(opacity=0.15), run_time=0.3)

        anchors = {
            "UL": np.array([-3.0,  1.4, 0]),
            "UR": np.array([ 3.0,  1.4, 0]),
            "DL": np.array([-3.0, -1.6, 0]),
            "DR": np.array([ 3.0, -1.6, 0]),
        }

        configs = [
            (-0.08, r"\delta<0,\ |\delta|\ \text{small}", "UL"),
            (-0.10, r"\delta<0,\ |\delta|\ \text{large}", "UR"),
            ( 0.08, r"\delta>0,\ \delta\ \text{small}", "DL"),
            ( 0.10, r"\delta>0,\ \delta\ \text{large}", "DR"),
        ]

        def make_spiral(delta, phi_max=8*np.pi, a=0.06, target=2.2):
            pts = log_spiral_points(delta, phi_max=phi_max, a=a)
            m = VMobject()
            m.set_points_smoothly(pts)
            m.set_stroke(width=6)

            # Rotar para que el extremo final apunte arriba
            P = m.get_points()
            p0 = P[0]
            p_end = P[-1]
            v = p_end - p0
            theta = angle_of_vector(v)
            m.rotate((PI/2) - theta)

            # Normalizar tamaño
            size = max(m.get_width(), m.get_height(), 1e-6)
            m.scale(target / size)
            return m

        spirals = VGroup()
        labels = VGroup()

        for delta, desc_tex, key in configs:
            sp = make_spiral(delta).move_to(anchors[key])
            lab = VGroup(
                Tex(r"\delta = %.3f" % delta).scale(0.62),
                Tex(desc_tex).scale(0.52),
            ).arrange(DOWN, buff=0.06).next_to(sp, DOWN, buff=0.18)
            spirals.add(sp)
            labels.add(lab)

        # Dibujar 4 + labels
        for sp, lab in zip(spirals, labels):
            self.play(ShowCreation(sp, rate_func=smooth), run_time=0.8)
            self.play(FadeIn(lab, shift=0.08*UP), run_time=0.25)

        # Beat corto: resaltar contraste
        self.play(spirals[0].animate.set_stroke(width=8), spirals[3].animate.set_stroke(width=8), run_time=0.35)
        self.play(spirals[0].animate.set_stroke(width=6), spirals[3].animate.set_stroke(width=6), run_time=0.35)

        self.wait(1.0)


###########################


class Fig03_PatternCusps(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.18)
        self.add(plane)

        title = Text("Fig. 3 — patrón de inversiones + cambios cuspidados").scale(0.55).to_edge(UP)
        self.add(title)

        curve = self.make_spiral_with_flip_pattern(
            # dominio
            x_min=-1.2,
            x_max=4.9,
            n=60000,

            # eventos de giro (tus “vueltas” discretas)
            turns_per_event=1.0,     # cuánto gira cada evento (≈ 1 vuelta)
            base_turn_rate=4.0,    # giro “de fondo” entre eventos #################################################### importante

            # schedule: invierte en estos índices de evento (1-indexed)
            # Ejemplo: invierte en 1, luego 3, luego 6, luego 10...
            flip_turns=list(range(10)),

            # spacing irregular entre eventos
            # Si pones una lista, define separaciones en x para cada evento.
            # Si None, usa period constante.
            period=0.48,
            period_pattern=[0.42, 0.55, 0.46, 0.62, 0.44, 0.58],  # se repite (rompe simetría)

            # cusp control (más chico = más “pico”)
            # edge_w controla el borde (transición) de los pulsos
            # cusp_sharpness refuerza el cruce abrupto de omega por 0 en los flips
            edge_w=0.010,
            pulse_width=0.22,
            cusp_sharpness=2.2,      # 1.0 suave, 2..3 mucho más “cusp-like”

            # estabilidad visual
            r_cut=0.0022,
            target_size=5.8,
            stroke_width=7,
            draw_center_out=True,

            # para evitar que el spline suavice demasiado los picos:
            use_corners=False,       # True => más “cúspide”, menos suave global
        )

        self.play(ShowCreation(curve), run_time=2.0)
        self.wait(1)

    # ------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------
    def make_spiral_with_flip_pattern(
        self,
        x_min=0,
        x_max=44.9,
        n=32000,
        turns_per_event=3.0,
        base_turn_rate=0.015,
        flip_turns=None,                 # list of ints (1-indexed)
        period=0.25,
        period_pattern=None,             # list of floats (repeating)
        edge_w=0.010,
        pulse_width=0.022,
        cusp_sharpness=20.2,
        r_cut=0.0022,
        target_size=5.8,
        stroke_width=7,
        draw_center_out=True,
        use_corners=True,
    ):
        if flip_turns is None:
            flip_turns = []

        x = np.linspace(x_min, x_max, n)
        dx = (x_max - x_min) / (n - 1)

        # r(x) = exp(-exp(x))
        r = np.exp(-np.exp(x))

        # recorte núcleo
        if r_cut is not None:
            mask = (r >= float(r_cut))
            x = x[mask]
            r = r[mask]
            n = len(x)
            dx = (x[-1] - x[0]) / max(n - 1, 1)

        # omega(x) = dθ/dx
        omega = 2 * np.pi * base_turn_rate * np.ones_like(x)

        # pulso base: para que el área del pulso sea 2π*turns_per_event
        omega_pulse_amp = 2 * np.pi * (turns_per_event / max(pulse_width, 1e-6))

        # ---- Construye tiempos de evento x_k con period irregular
        xk = []
        cur = x_min
        k = 0
        patt = period_pattern[:] if period_pattern else None

        while cur < x_max + 2 * period:
            xk.append(cur)
            if patt:
                cur += patt[k % len(patt)]
            else:
                cur += period
            k += 1

        # ---- Aplica pulsos con signo que puede invertirse según flip_turns
        # sign_k se mantiene hasta que ocurre un flip.
        sgn = +1
        flip_set = set(int(m) for m in flip_turns)  # eventos 1-indexed

        for idx, x0 in enumerate(xk, start=1):
            # ventana del evento
            xL = x0
            xR = x0 + pulse_width

            w = self._cusp_window(x, xL, xR, edge_w, cusp_sharpness)

            # pulso principal
            omega += sgn * omega_pulse_amp * w

            # si este evento es flip, fuerza un cruce más “agudo” por 0
            if idx in flip_set:
                # "cusp booster": añade un pico antisímétrico para que omega cambie de signo con kink visible
                # Es básicamente una derivada localizada (forma de “punta”) controlada por cusp_sharpness.
                booster = cusp_sharpness * omega_pulse_amp * self._cusp_booster(x, xL, xR)
                omega += (-sgn) * booster

                # y ahora sí invierte el sentido para los eventos siguientes
                sgn *= -1

        # integrar
        theta = np.cumsum(omega) * dx

        pts = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(r)], axis=1)

        # centro→afuera
        if draw_center_out:
            pts = pts[::-1]

        m = VMobject()
        if use_corners:
            # más “cúspide”, menos suave global
            m.set_points_as_corners(pts)
        else:
            #m.set_points_smoothly(pts)
            m.set_points_as_corners(pts)

        m.set_stroke(WHITE, width=stroke_width)

        # normaliza y orienta
        m.move_to(ORIGIN)
        size = max(m.get_width(), m.get_height(), 1e-3)
        m.scale(target_size / size)
        m.move_to(ORIGIN)

        end = m.get_points()[-1]
        m.rotate((PI/4) - angle_of_vector(end - ORIGIN))

        return m

    # ------------------------------------------------------------
    # “Cusp-like” windows
    # ------------------------------------------------------------
    def _smooth_step(self, x, x0, w):
        w = max(float(w), 1e-6)
        return  0.5* (1.0 + np.tanh((x - x0) / w))

    def _cusp_window(self, x, xL, xR, w, sharp):
        """
        Ventana con bordes más duros que tanh:
        tomamos ventana suave y la 'endurecemos' elevando a una potencia.
        sharp > 1 => bordes más verticales (más cusp-like)
        """
        base = self._smooth_step(x, xL, w) - self._smooth_step(x, xR, w)
        base = np.clip(base, 0.0, 1.0)
        p = max(1.0, float(sharp))
        return base**p

    def _cusp_booster(self, x, xL, xR):
        """
        Pulso antisímétrico localizado: positivo cerca de xL y negativo cerca de xR,
        empuja omega a cruzar 0 de manera más abrupta.
        """
        mid = 0.5 * (xL + xR)
        width = max(0.18 * (xR - xL), 1e-6)
        # diferencia de dos gaussianas (antisímétrica)
        g1 = np.exp(-((x - (mid - 0.25*(xR-xL)))**2) / (2*width**2))
        g2 = np.exp(-((x - (mid + 0.25*(xR-xL)))**2) / (2*width**2))
        return (g1 - g2)

###########################################################################
# -------------------------
# Helpers (Fig. 4) — safe envelopes (no crossings between turns)
# -------------------------
from manimlib import *
import numpy as np


def _bounded_wavy_spiral_points(
    x_min=-1.0,
    x_max=5.2,
    n=60000,
    base_turn_rate=2.8,
    A_env=0.10,              # envelope thickness parameter
    a_wave=0.06,             # wave amplitude INSIDE envelopes (must be < A_env)
    m=10,
    phase=0.0,
    r_cut=0.0018,
    safety=0.92,             # shrink factor so we're strictly safe
):
    """
    Builds:
      - main wavy spiral r(x)*(1 + a_wave*sin(m*theta+phase))
      - envelopes r(x)*(1 ± A_env)
    but clamps A_env to avoid crossings between envelopes across different turns.
    """

    x = np.linspace(x_min, x_max, n)
    dx = (x_max - x_min) / (n - 1)

    # Base radius
    r = np.exp(-np.exp(x))

    if r_cut is not None:
        mask = (r >= float(r_cut))
        x, r = x[mask], r[mask]
        n = len(x)
        dx = (x[-1] - x[0]) / max(n - 1, 1)

    # theta'(x) = 2π * base_turn_rate
    omega = 2 * np.pi * float(base_turn_rate) * np.ones_like(x)
    theta = np.cumsum(omega) * dx

    # ------------------------------------------------------------
    # Compute A_env_max so envelopes do NOT cross between turns
    # Condition per same angle mod 2π:
    #   (1 + A) r_inner < (1 - A) r_outer
    # Let q = r_inner / r_outer  (q < 1)  =>  A < (1 - q)/(1 + q)
    #
    # Since theta is linear in x, one turn corresponds to Δx = 1/base_turn_rate.
    # We approximate q(x) = r(x) / r(x - Δx) for x where x-Δx is in range.
    # ------------------------------------------------------------
    base_turn_rate = float(base_turn_rate)
    dx_turn = 1.0 / max(base_turn_rate, 1e-9)  # x increment per 2π

    # Interpolate r(x - dx_turn)
    x_prev = x - dx_turn
    valid = (x_prev >= x[0])
    if np.any(valid):
        r_prev = np.interp(x_prev[valid], x, r)  # r at previous turn (outer)
        q = np.clip(r[valid] / np.maximum(r_prev, 1e-30), 0.0, 0.999999)
        A_max_local = (1.0 - q) / (1.0 + q)       # per-sample bound
        A_env_max = float(np.min(A_max_local))    # global safe bound
        A_env_max = max(0.0, min(0.95, safety * A_env_max))
    else:
        # if range is < one full turn, no cross-turn issue
        A_env_max = 0.95

    A_env_req = float(A_env)
    A_env = min(max(A_env_req, 0.0), A_env_max)

    # Wave amplitude strictly inside envelopes
    a_wave_req = float(a_wave)
    a_wave = min(max(a_wave_req, 0.0), 0.98 * max(A_env, 0.0))

    # Main curve radius (guaranteed between envelopes for same parameter)
    r_wavy = r * (1.0 + a_wave * np.sin(float(m) * theta + float(phase)))

    # Points
    pts = np.stack([r_wavy * np.cos(theta), r_wavy * np.sin(theta), np.zeros_like(r)], axis=1)
    pts_up = np.stack([(r * (1.0 + A_env)) * np.cos(theta), (r * (1.0 + A_env)) * np.sin(theta), np.zeros_like(r)], axis=1)
    pts_dn = np.stack([(r * (1.0 - A_env)) * np.cos(theta), (r * (1.0 - A_env)) * np.sin(theta), np.zeros_like(r)], axis=1)

    meta = dict(A_env_used=A_env, A_env_max=A_env_max, a_wave_used=a_wave)
    return pts, pts_up, pts_dn, meta


class Fig04_CompositeWavyEnvelopes(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.18)
        self.add(plane)

        title = Text("Fig. 4 — composite spiral (bounded wavy envelopes)") \
            .scale(0.52).to_edge(UP)
        self.add(title)

        # -------------------------
        # Tunables
        # -------------------------
        base_turn_rate = 2.6
        m = 9
        phase = 0.5

        A_env = 0.05     # desired envelope thickness (will be auto-clamped if too big)
        a_wave = 0.075   # wave amplitude INSIDE envelopes (should be < A_env)

        stroke_curve = 2.2      # thinner spiral
        stroke_env = 2.6
        env_opacity = 0.95
        env_up_color = BLUE
        env_dn_color = RED

        target_size = 6.2
        tilt = 0.35

        pts, pts_up, pts_dn, meta = _bounded_wavy_spiral_points(
            x_min=-1.1,
            x_max=5.4,
            n=24000,
            base_turn_rate=base_turn_rate,
            A_env=A_env,
            a_wave=a_wave,
            m=m,
            phase=phase,
            r_cut=0.0018,
            safety=0.92,
        )

        # Optional: show the effective parameters (debug)
        dbg = Text(
            f"A_env used={meta['A_env_used']:.3f} (max={meta['A_env_max']:.3f}),  a_wave={meta['a_wave_used']:.3f}",
            font_size=22
        ).to_corner(DR)
        dbg.set_opacity(0.7)
        self.add(dbg)

        # Main curve
        curve = VMobject()
        curve.set_points_as_corners(pts[::-1])  # center -> out
        curve.set_stroke(WHITE, width=stroke_curve)

        # Envelopes (continuous, colored)
        env_up = VMobject().set_points_smoothly(pts_up[::-1])
        env_dn = VMobject().set_points_smoothly(pts_dn[::-1])
        env_up.set_stroke(env_up_color, width=stroke_env, opacity=env_opacity)
        env_dn.set_stroke(env_dn_color, width=stroke_env, opacity=env_opacity)

        # Normalize size + orientation
        group = VGroup(env_up, env_dn, curve).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        # Animation
        self.play(ShowCreation(env_up), ShowCreation(env_dn), run_time=1.0)
        self.play(ShowCreation(curve), run_time=2.0)
        self.wait(1.0)

        self.play(curve.animate.set_stroke(width=stroke_curve + 1.2), run_time=0.20)
        self.play(curve.animate.set_stroke(width=stroke_curve), run_time=0.20)
        self.wait(1.0)

#####################################################
from manimlib import *
import numpy as np

TAU = 2 * np.pi


def _select_by_turn_window(turn_count, t0, t1, min_points=8):
    """
    Return indices where t0 <= turn_count <= t1.
    Robust to t0>t1 (swap).
    """
    a = float(min(t0, t1))
    b = float(max(t0, t1))
    idx = np.where((turn_count >= a) & (turn_count <= b))[0]
    if len(idx) < min_points:
        raise ValueError(
            f"Turn window too small or out of range: [{a:.3f}, {b:.3f}] "
            f"-> only {len(idx)} points. Increase n/x_max or widen window."
        )
    return idx


def _spiral_precompute(
    x_min=-1.05,
    x_max=5.10,
    n=26000,
    base_turn_rate=0.78,
    r_cut=0.0018,
):
    """
    Precompute the master parameter arrays:
      x, r(x), theta(x), turn_count(x)
    """
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


def build_spiral_band_with_independent_windows(
    x_min=-1.05,
    x_max=5.10,
    n=26000,
    base_turn_rate=0.78,
    r_cut=0.0018,
    band_w=0.24,
    # Independent intervals in "turn space"
    center_turn_window=None,        # e.g. (0.0, 4.5) or None for full available
    outer_turn_window=(0.0, 3.85),
    inner_turn_window=(0.0, 4.35),
):
    """
    Returns:
      pts_center, pts_outer, pts_inner, meta

    Each border uses its own (start_turn, end_turn) window.
    """
    x, r, theta, turn_count = _spiral_precompute(
        x_min=x_min, x_max=x_max, n=n,
        base_turn_rate=base_turn_rate,
        r_cut=r_cut
    )

    w = float(np.clip(band_w, 0.02, 0.49))
    r_c = r
    r_o = r * (1.0 + w)
    r_i = r * (1.0 - w)

    # Windows
    if center_turn_window is None:
        idx_c = np.arange(len(x))
    else:
        idx_c = _select_by_turn_window(turn_count, center_turn_window[0], center_turn_window[1])

    idx_o = _select_by_turn_window(turn_count, outer_turn_window[0], outer_turn_window[1])
    idx_i = _select_by_turn_window(turn_count, inner_turn_window[0], inner_turn_window[1])

    pts_c = _to_pts(r_c[idx_c], theta[idx_c])
    pts_o = _to_pts(r_o[idx_o], theta[idx_o])
    pts_i = _to_pts(r_i[idx_i], theta[idx_i])

    meta = dict(
        x=x,
        r=r,
        theta=theta,
        turn_count=turn_count,
        band_w=w,
        idx_c=idx_c,
        idx_o=idx_o,
        idx_i=idx_i,
        outer_turn_window=outer_turn_window,
        inner_turn_window=inner_turn_window,
        center_turn_window=center_turn_window,
    )
    return pts_c, pts_o, pts_i, meta


def _mini_log_spiral_points(
    turns=1.10,
    n=220,
    a=0.018,
    growth=0.28,
    clockwise=True,
    flatten=0.85,
):
    sgn = -1.0 if clockwise else 1.0
    t = np.linspace(0.0, TAU * turns, n)
    rho = a * np.exp(growth * t)
    x = rho * np.cos(sgn * t)
    y = flatten * rho * np.sin(sgn * t)
    return np.stack([x, y, np.zeros_like(x)], axis=1)


def _tangent_angle(pts, k):
    k0 = int(np.clip(k, 1, len(pts) - 2))
    v = pts[k0 + 1] - pts[k0 - 1]
    return angle_of_vector(v)


class Fig05_ScrollBand(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.12)
        self.add(plane)

        title = Text("Fig. 5 — independent windows (outer/inner)").scale(0.58).to_edge(UP)
        self.add(title)

        # -------------------------
        # Tunables
        # -------------------------
        base_turn_rate = 0.95
        band_w = 0.24
        stroke_band = 4.0
        stroke_orn = 3.0
        target_size = 6.25
        tilt = 0.30

        # Independent turn windows (EDIT THESE)
        # You can move each curve's interval freely:
        outer_turn_window = (0.40, 27.20) #(0.20, 3.85)   # start, end (turns)
        inner_turn_window = (0.20, 3.85) #(0.40, 6.20)   # inner can start later and end later
        center_turn_window = (0.15, 4.60)  # used for ornaments / tangents

        ornaments_per_turn = 3
        turns_for_ornaments = 4.20
        outer_big = False
        inner_small = False

        # -------------------------
        # Geometry: center + independent borders
        # -------------------------
        pts_c, pts_o, pts_i, meta = build_spiral_band_with_independent_windows(
            x_min=-1.05,
            x_max=6.10,
            n=26000,
            base_turn_rate=base_turn_rate,
            r_cut=0.0018,
            band_w=band_w,
            center_turn_window=center_turn_window,
            outer_turn_window=outer_turn_window,
            inner_turn_window=inner_turn_window,
        )

        # Reverse to draw center -> outward (optional)
        pts_c = pts_c[::-1]
        pts_o = pts_o[::-1]
        pts_i = pts_i[::-1]

        # For ornaments, recompute turn_count along the chosen center window
        # Use meta arrays filtered by idx_c, then reverse consistently.
        idx_c = meta["idx_c"]
        theta_c = meta["theta"][idx_c][::-1]
        r_c = meta["r"][idx_c][::-1]
        turn_count_c = (theta_c - theta_c[0]) / TAU

        outer = VMobject().set_points_smoothly(pts_o)
        inner = VMobject().set_points_smoothly(pts_i)
        outer.set_stroke(WHITE, width=stroke_band, opacity=0.95)
        inner.set_stroke(WHITE, width=stroke_band, opacity=0.95)

        # -------------------------
        # Ornaments (optional)
        # -------------------------
        ornaments = VGroup()

        max_turn = float(min(turns_for_ornaments, turn_count_c[-1]))
        step = 1.0 / max(1, int(ornaments_per_turn))
        targets = np.arange(0.8, max_turn, step)

        idxs = []
        last_k = -10**9
        min_sep = 220

        # Now we index into pts_c (already reversed) using turn_count_c
        for t in targets:
            k = int(np.argmin(np.abs(turn_count_c - t)))
            if (k - last_k) > min_sep:
                idxs.append(k)
                last_k = k

        base_sp = _mini_log_spiral_points(turns=1.10, n=220, a=0.018, growth=0.28, clockwise=True)

        for j, k in enumerate(idxs):
            k = int(np.clip(k, 1, len(pts_c) - 2))
            p = pts_c[k]
            ang = _tangent_angle(pts_c, k)

            s = 0.35 + 0.11 * min(turn_count_c[k], max_turn)

            tvec = pts_c[k + 1] - pts_c[k - 1]
            tvec = tvec / (np.linalg.norm(tvec) + 1e-9)
            nvec = rotate_vector(tvec, PI / 2)

            w_local = band_w * r_c[k]

            if outer_big:
                sp = VMobject().set_points_smoothly(base_sp.copy())
                sp.rotate(ang + 0.20)
                sp.scale(1.25 * s)
                sp.shift(p + (0.85 * w_local) * nvec)
                sp.set_stroke(WHITE, width=stroke_orn, opacity=0.95)
                ornaments.add(sp)

            if inner_small and (j % 2 == 0):
                sp2 = VMobject().set_points_smoothly(base_sp.copy())
                sp2.rotate(ang - 0.35)
                sp2.scale(0.70 * s)
                sp2.shift(p - (0.55 * w_local) * nvec)
                sp2.set_stroke(WHITE, width=stroke_orn * 0.95, opacity=0.95)
                ornaments.add(sp2)

        # -------------------------
        # Normalize + tilt
        # -------------------------
        band_only = VGroup(outer, inner).move_to(ORIGIN)
        ref_size = max(band_only.get_width(), band_only.get_height(), 1e-3)

        group = VGroup(outer, inner, ornaments).move_to(ORIGIN)
        group.scale(target_size / ref_size).move_to(ORIGIN)
        group.rotate(tilt)

        # -------------------------
        # Animate
        # -------------------------
        self.play(ShowCreation(outer), ShowCreation(inner), run_time=1.2)
        self.wait(1.0)


#########################################################################3
###########################################################################

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
# 2) Glue (tu lógica, con un micro-fix de robustez)
# ============================================================
def _tangent_vec(pts, at_start=True):
    if len(pts) < 3:
        return RIGHT
    v = (pts[1] - pts[0]) if at_start else (pts[-1] - pts[-2])
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _rotate_pts(pts, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]])
    return pts @ R.T


def glue_two_bands(
    A_chain, B_chain,
    glue_key="o",          # "o" (outer) or "i" (inner) or "c" (center)
    use_end_of_A=True,     # glue at A end (True) or A start (False)
    use_start_of_B=True,   # glue B start (True) or B end (False)
    flip_B=True,           # reverse B curves BEFORE aligning (scroll-like)
    angle_bias=0.0,
):
    """
    Align B so that B(glue_key, anchor) matches A(glue_key, anchor) in position and tangent.
    Applies same rigid transform to B["c"], B["o"], B["i"].
    """
    # reverse B direction if desired
    if flip_B:
        B_chain = {k: v[::-1].copy() for k, v in B_chain.items()}

    A_pts = A_chain[glue_key]
    B_pts = B_chain[glue_key]

    # anchors + tangents
    A_anchor = A_pts[-1] if use_end_of_A else A_pts[0]
    A_tan = _tangent_vec(A_pts, at_start=not use_end_of_A)

    B_anchor = B_pts[0] if use_start_of_B else B_pts[-1]
    B_tan = _tangent_vec(B_pts, at_start=use_start_of_B)

    angA = angle_of_vector(A_tan)
    angB = angle_of_vector(B_tan)
    rot = (angA - angB) + float(angle_bias)

    out = {}
    for key, pts in B_chain.items():
        pr = _rotate_pts(pts, rot)
        B_anchor_rot = pr[0] if use_start_of_B else pr[-1]
        shift = A_anchor - B_anchor_rot
        out[key] = pr + shift

    return out


# ============================================================
# 3) Scene
# ============================================================
class TwoSpiralBand_Glued(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.12)
        self.add(plane)

        title = Text("Two spiral bands glued (outer-driven)").scale(0.58).to_edge(UP)
        self.add(title)

        stroke_band = 4.0
        target_size = 6.3
        tilt = 0.20

        # --- Build A ---
        A_pts, _ = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            center_turn_window=(0.15, 4.60),
            outer_turn_window=(0.40, 4.20),
            inner_turn_window=(0.20, 4.70),
            reverse=True,
        )

        # --- Build B (local) ---
        B_pts, _ = build_band(
            x_min=-1.05, x_max=6.10, n=26000,
            base_turn_rate=0.95, r_cut=0.0018,
            band_w=0.24,
            center_turn_window=(0.15, 4.60),
            outer_turn_window=(0.30, 4.10),
            inner_turn_window=(0.10, 4.60),
            reverse=True,
        )

        # --- Glue using OUTER ---
        B_pts = glue_two_bands(
            A_chain=A_pts,
            B_chain=B_pts,
            glue_key="o",
            use_end_of_A=True,
            use_start_of_B=True,
            flip_B=True,
            angle_bias=0.0,
        )

        A_outer, A_inner = make_band_mobjects(A_pts, stroke_band=stroke_band, color=WHITE)
        B_outer, B_inner = make_band_mobjects(B_pts, stroke_band=stroke_band, color=WHITE)

        group = VGroup(A_outer, A_inner, B_outer, B_inner).move_to(ORIGIN)
        size = max(group.get_width(), group.get_height(), 1e-3)
        group.scale(target_size / size).move_to(ORIGIN)
        group.rotate(tilt)

        # Debug: glue point (outer end of A after transforms is not directly A_outer.get_end()
        # because we moved/scaled/rotated the whole group; easiest is to place it after transform:
        # We'll compute it from the mobject after all transforms.
        self.add(Dot(A_outer.get_end(), radius=0.06).set_color(YELLOW))

        self.play(ShowCreation(A_outer), ShowCreation(A_inner), run_time=1.0)
        #self.play(ShowCreation(B_outer), ShowCreation(B_inner), run_time=1.0)
        self.wait(1.0)
##################################################################################################33


