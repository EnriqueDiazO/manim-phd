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
        self.wait(2.0)
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
            (-0.10, r"\delta<0,\ |\delta|\ \text{small}", "UL"),
            (-0.08, r"\delta<0,\ |\delta|\ \text{large}", "UR"),
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

###############3