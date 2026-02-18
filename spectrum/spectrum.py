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



###############################3
# -------------------------
# Fig. 3–4 (oscillatory spiral variants) — approximate
# -------------------------

from manimlib import *
import numpy as np

class Figura3_Simple(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        # Título simple
        title = Text("Fig. 3", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.add(title)
        
        # Crear la curva de la manera más simple posible
        curva = self.crear_curva_simple()
        self.add(curva)
        self.wait(2)
    
    def crear_curva_simple(self):
        # Parámetros para una espiral logarítmica básica
        t = np.linspace(0, 6, 8000)
        r = np.exp(-t)
        theta = 0.3 * t * 2 * np.pi  # Controla el número de vueltas
        
        # Puntos
        puntos = []
        for i in range(len(t)):
            puntos.append([
                r[i] * np.cos(theta[i]),
                r[i] * np.sin(theta[i]),
                0
            ])
        
        # Crear curva usando el método más básico
        curva = VMobject()
        curva.set_points_as_corners(puntos)
        curva.set_stroke(color=WHITE, width=2)
        
        # Centrar y escalar
        curva.center()
        curva.scale(4)
        
        return curva


class Figura3_Compuesta(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        title = Text("Fig. 3", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.add(title)
        
        # Crear grupo de curvas
        grupo = self.crear_grupo_espirales()
        self.add(grupo)
        self.wait(2)
    
    def crear_grupo_espirales(self):
        grupo = VGroup()
        
        # Rango para todas las espirales
        t = np.linspace(0, 6, 6000)
        r = np.exp(-t)
        
        # Espiral principal (más gruesa)
        theta1 = 0.3 * t * 2 * np.pi
        puntos1 = self.generar_puntos(r, theta1)
        curva1 = self.crear_curva_desde_puntos(puntos1, WHITE, 3)
        grupo.add(curva1)
        
        # Espirales secundarias con diferentes parámetros
        configs = [
            (0.28, 0.5, interpolate_color(WHITE, BLUE_E, 0.2), 1.5),
            (0.32, 1.2, interpolate_color(WHITE, BLUE_E, 0.3), 1.2),
            (0.26, 2.8, interpolate_color(WHITE, BLUE_E, 0.4), 1.0),
        ]
        
        for delta, offset, color, grosor in configs:
            theta = delta * t * 2 * np.pi + offset
            puntos = self.generar_puntos(r, theta)
            curva = self.crear_curva_desde_puntos(puntos, color, grosor)
            grupo.add(curva)
        
        grupo.center()
        grupo.scale(4)
        
        return grupo
    
    def generar_puntos(self, r, theta):
        puntos = []
        for i in range(len(r)):
            puntos.append([
                r[i] * np.cos(theta[i]),
                r[i] * np.sin(theta[i]),
                0
            ])
        return puntos
    
    def crear_curva_desde_puntos(self, puntos, color, grosor):
        curva = VMobject()
        curva.set_points_as_corners(puntos)
        curva.set_stroke(color=color, width=grosor)
        return curva


class ComposicionFinal(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        # Título de autores
        autores = Text("A. Böttcher and Yu. I. Karlovich", font_size=36, color=WHITE)
        autores.to_edge(UP)
        self.add(autores)
        
        # Figura 3
        fig3 = self.crear_figura_3()
        fig3.move_to(LEFT * 4)
        
        # Figura 4
        fig4 = self.crear_figura_4()
        fig4.move_to(RIGHT * 4)
        
        # Etiquetas
        label3 = Text("Fig. 3", font_size=30, color=WHITE)
        label3.next_to(fig3, DOWN, buff=0.5)
        
        label4 = Text("Fig. 4", font_size=30, color=WHITE)
        label4.next_to(fig4, DOWN, buff=0.5)
        
        # Añadir todo
        self.add(fig3, fig4, label3, label4)
        self.wait(2)
    
    def crear_figura_3(self):
        grupo = VGroup()
        
        t = np.linspace(0, 6, 6000)
        r = np.exp(-t)
        
        # Espiral principal
        theta_main = 0.3 * t * 8 * np.pi
        puntos_main = self.generar_puntos(r, theta_main)
        curva_main = self.crear_curva(puntos_main, WHITE, 3)
        grupo.add(curva_main)
        
        # Secundarias
        for delta, offset in [(0.28, 0.5), (0.32, 1.2)]:
            theta = delta * t * 8 * np.pi + offset
            puntos = self.generar_puntos(r, theta)
            curva = self.crear_curva(puntos, interpolate_color(WHITE, BLUE_E, 0.2), 1.5)
            grupo.add(curva)
        
        grupo.center()
        grupo.scale(4)
        return grupo
    
    def crear_figura_4(self):
        grupo = VGroup()
        
        t = np.linspace(0, 7, 8000)
        r = np.exp(-t)
        
        # Espiral principal
        theta_main = 0.32 * t * 2 * np.pi
        puntos_main = self.generar_puntos(r, theta_main)
        curva_main = self.crear_curva(puntos_main, WHITE, 3)
        grupo.add(curva_main)
        
        # Múltiples secundarias con más variación
        for i, (delta, offset) in enumerate([
            (0.28, 0.3), (0.34, 1.1), (0.30, 1.9), 
            (0.36, 2.7), (0.26, 3.5)
        ]):
            theta = delta * t * 2 * np.pi + offset
            puntos = self.generar_puntos(r, theta)
            color = interpolate_color(WHITE, BLUE_E, 0.1 * i)
            curva = self.crear_curva(puntos, color, 1.5 - i * 0.2)
            grupo.add(curva)
        
        grupo.center()
        grupo.scale(4)
        return grupo
    
    def generar_puntos(self, r, theta):
        puntos = []
        for i in range(len(r)):
            puntos.append([r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i]), 0])
        return puntos
    
    def crear_curva(self, puntos, color, grosor):
        curva = VMobject()
        curva.set_points_as_corners(puntos)
        curva.set_stroke(color=color, width=grosor)
        return curva


# Para probar
class TestSimple(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        self.add(Text("Test", font_size=72))
        self.wait(1)

class TestEspiralSimple(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        
        # Espiral simple
        t = np.linspace(0, 4*np.pi, 1000)
        r = np.exp(-0.3 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        puntos = []
        for i in range(len(t)):
            puntos.append([x[i], y[i], 0])
        
        curva = VMobject()
        curva.set_points_as_corners(puntos)
        curva.set_stroke(WHITE, width=2)
        curva.center()
        curva.scale(3)
        
        self.add(curva)
        self.wait(2)


#######################


class Fig03_04_OscillatorySpiral(Scene):
    def construct(self):
        plane = NumberPlane()
        plane.set_stroke(width=1, opacity=0.20)
        self.add(plane)

        # ========== Parámetros “tipo paper” ==========
        # Ajusta SOLO estos tres primero:
        delta = 0.18
        lam   = 6.0
        gam   = 0.25

        # Control visual: cuántas vueltas quieres en Fig. 3
        # (Yuri tiene ~8–12 "anillos" visibles)
        turns_target = 10

        # rmax cercano a 1 para que termine en el borde exterior
        rmax = 0.98

        # rmin se calcula para aproximar "turns_target"
        rmin = self.pick_rmin_for_turns(delta=delta, rmax=rmax, turns_target=turns_target)

        title = Text("Fig. 3–4 (paper):  $\\gamma(r)=r e^{i\\theta(r)}$,  $\\theta(r)=h(\\log|\\log r|)|\\log r|$").scale(0.45).to_edge(UP)
        self.play(FadeIn(title), run_time=0.5)

        # ========== Fig. 3: h(x) ≡ δ ==========
        fig3 = self.paper_spiral(delta=delta, gamma=0.0, lam=lam, rmin=rmin, rmax=rmax, n=9000)
        fig3.set_stroke(width=4)
        fig3 = self.fit_to_box(fig3, target_w=5.5, target_h=3.3).shift(UP*1.75)

        label3 = Text("Fig. 3").scale(0.45).next_to(fig3, RIGHT, buff=0.5)

        # ========== Fig. 4: h(x)=δ+γ sin(λx) + envolventes δ±γ ==========
        fig4 = self.paper_spiral(delta=delta, gamma=gam, lam=lam, rmin=rmin, rmax=rmax, n=9000)
        fig4.set_stroke(width=4)

        env_plus  = self.paper_spiral(delta=delta+gam, gamma=0.0, lam=lam, rmin=rmin, rmax=rmax, n=6000).set_stroke(width=2, opacity=0.8)
        env_minus = self.paper_spiral(delta=delta-gam, gamma=0.0, lam=lam, rmin=rmin, rmax=rmax, n=6000).set_stroke(width=2, opacity=0.8)

        env_plus  = DashedVMobject(env_plus,  num_dashes=90)
        env_minus = DashedVMobject(env_minus, num_dashes=90)

        # Encajar a misma caja que fig4 y bajar
        fig4 = self.fit_to_box(fig4, target_w=5.5, target_h=3.3).shift(DOWN*1.75)
        env_plus  = self.fit_to_box(env_plus,  target_w=5.5, target_h=3.3).move_to(fig4.get_center())
        env_minus = self.fit_to_box(env_minus, target_w=5.5, target_h=3.3).move_to(fig4.get_center())

        label4 = Text("Fig. 4").scale(0.45).next_to(fig4, RIGHT, buff=0.5)

        # ========== Animación ==========
        self.play(ShowCreation(fig3), FadeIn(label3), run_time=1.2)
        self.wait(0.2)

        self.play(ShowCreation(env_plus), ShowCreation(env_minus), run_time=0.9)
        self.play(ShowCreation(fig4), FadeIn(label4), run_time=1.2)
        self.wait(1.0)

    # ------------------------------------------------------------
    # Curva EXACTA del paper:
    #   γ(r)= r e^{iθ(r)},  θ(r)=h(log|log r|)*|log r|
    #   h(x)=δ + γ sin(λx)
    # ------------------------------------------------------------
    def paper_spiral(self, delta, gamma, lam, rmin, rmax, n=8000):
        r = np.linspace(rmin, rmax, n)

        L = np.abs(np.log(r))             # |log r|
        x = np.log(np.abs(np.log(r)))     # log|log r|
        h = delta + gamma*np.sin(lam*x)
        theta = h * L

        z = r * np.exp(1j*theta)
        pts = np.stack([z.real, z.imag, np.zeros_like(z.real)], axis=1)

        m = VMobject()
        m.set_points_smoothly(pts)

        # Rotación: fija el remate exterior hacia arriba (consistencia visual)
        P = m.get_points()
        v = P[-1] - P[0]
        ang = angle_of_vector(v)
        m.rotate((PI/2) - ang)
        return m

    # ------------------------------------------------------------
    # Elegir rmin para obtener ~turns_target vueltas en el caso γ=0.
    # Aproximación: número de vueltas ≈ (θ(rmin)-θ(rmax)) / (2π)
    # con θ(r)=δ|log r|
    # => turns ≈ δ(|log rmin|-|log rmax|)/(2π)
    # ------------------------------------------------------------
    def pick_rmin_for_turns(self, delta, rmax, turns_target):
        Lmax = np.abs(np.log(rmax))
        Lmin = Lmax + (2*np.pi*turns_target)/max(delta, 1e-6)
        # rmin = exp(-Lmin)
        rmin = float(np.exp(-Lmin))
        # seguridad numérica: evita underflow extremo
        return max(rmin, 1e-50)

    def fit_to_box(self, mobj, target_w=6.0, target_h=3.0):
        w = max(mobj.get_width(), 1e-6)
        h = max(mobj.get_height(), 1e-6)
        s = min(target_w / w, target_h / h)
        return mobj.copy().scale(s)




# -------------------------
# Fig. 5–6 (compound arcs) — approximate stamping spirals
# -------------------------

class Fig05_06_CompoundArc(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        # Backbone curve (simple smooth arc)
        def backbone(t):
            return np.array([6*(t-0.5), 1.5*np.sin(2*np.pi*t), 0.0])

        back = ParametricCurve(backbone, t_range=[0, 1, 0.01]).set_stroke(width=6)
        self.play(ShowCreation(back))
        self.wait(0.3)

        # Stamp small spirals along it
        ts = np.linspace(0.1, 0.9, 7)
        motifs = VGroup()
        for k, t in enumerate(ts):
            p = backbone(t)
            delta = 0.18 if (k % 2 == 0) else -0.12
            pts = log_spiral_points(delta, phi_max=6*np.pi, a=0.02)
            sp = VMobject().set_points_smoothly(pts).set_stroke(width=3)
            sp.scale(1.2)
            sp.shift(p)
            motifs.add(sp)

        for sp in motifs:
            self.play(ShowCreation(sp), run_time=0.25)

        self.wait(1)

# -------------------------
# Fig. 7 (power weight zeros/poles schematic)
# -------------------------

class Fig07_PowerWeightZerosPoles(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        curve = Line(LEFT*5, RIGHT*5).set_stroke(width=6)
        self.play(ShowCreation(curve))

        pts = [LEFT*3, ORIGIN, RIGHT*3]
        mus = [0.7, -0.7, 0.9]  # + => zero, - => pole (schematic)
        dots = VGroup(*[Dot(p, radius=0.08) for p in pts])

        labels = VGroup()
        for p, mu in zip(pts, mus):
            if mu > 0:
                txt = Text(f"μ={mu:+.1f} (zero)").scale(0.45).next_to(p, UP)
            else:
                txt = Text(f"μ={mu:+.1f} (pole)").scale(0.45).next_to(p, DOWN)
            labels.add(txt)

        self.play(FadeIn(dots), FadeIn(labels))
        self.wait(1)

# -------------------------
# Fig. 8 (oscillating weight along an arc) — thickness modulation
# -------------------------

class Fig08_OscillatingWeight(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        lam = 6.0
        delta_vals = np.linspace(0.0, 1.2, 60)

        def base_arc(t):
            # arc in plane
            x = -5 + 10*t
            y = 1.5*np.sin(2*np.pi*t)
            return np.array([x, y, 0.0])

        arc = ParametricCurve(base_arc, t_range=[0, 1, 0.01]).set_stroke(width=4)
        self.play(ShowCreation(arc))
        title = Text("Oscillating weight (thickness ∝ exp(δ sin(λ s)))").scale(0.45).to_corner(UL)
        self.add(title)

        moving = arc.copy()

        for delta in delta_vals:
            new = self._weighted_arc(base_arc, lam=lam, delta=delta)
            self.play(Transform(moving, new), run_time=0.05)

        self.wait(1)

    def _weighted_arc(self, base_arc, lam, delta, n=400):
        ts = np.linspace(0, 1, n)
        pts = np.array([base_arc(t) for t in ts])
        # thickness samples: w = exp(delta sin(lam t))
        w = np.exp(delta*np.sin(lam*ts))
        # ManimGL stroke width is global; approximate by splitting into segments
        segs = VGroup()
        for i in range(n-1):
            width = 1.0 + 4.0*(w[i] / (w.max() + 1e-9))
            seg = Line(pts[i], pts[i+1]).set_stroke(width=width)
            segs.add(seg)
        return segs

# -------------------------
# Fig. 12–14 (arc → horn metamorphosis)
# -------------------------

class Fig12_14_ArcToHorn(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        pm = VGroup(Dot(LEFT, radius=0.07), Dot(RIGHT, radius=0.07))
        self.add(pm)

        c1 = 0.3
        c2_vals = np.linspace(0.35, 1.2, 60)

        arc1 = VMobject().set_points_smoothly(circle_arc_between_pm1(c1)).set_stroke(width=6)
        arc1_label = Text("arc").scale(0.55).next_to(arc1, UP)

        self.play(ShowCreation(arc1), FadeIn(arc1_label))
        self.wait(0.3)

        arc2 = VMobject().set_points_smoothly(circle_arc_between_pm1(c1+0.05)).set_stroke(width=4).set_color(YELLOW)
        horn_fill = VMobject()

        self.play(ShowCreation(arc2))
        self.wait(0.2)

        for c2 in c2_vals:
            new_arc2 = VMobject().set_points_smoothly(circle_arc_between_pm1(c2)).set_stroke(width=4).set_color(YELLOW)

            # fill between arcs (approx polygon)
            pts = np.vstack([
                circle_arc_between_pm1(c1),
                circle_arc_between_pm1(c2)[::-1],
            ])
            new_fill = Polygon(*pts).set_fill(opacity=0.25).set_stroke(width=0)

            self.play(Transform(arc2, new_arc2), Transform(horn_fill, new_fill), run_time=0.05)

        horn_label = Text("horn (region between arcs)").scale(0.55).to_corner(UR)
        self.play(FadeIn(horn_label))
        self.wait(1)

# -------------------------
# Fig. 18 (transform pipeline): indicator → intermediate → leaf
# -------------------------

class Fig18_TransformPipeline(Scene):
    def construct(self):
        # 3 panels
        left = NumberPlane().scale(0.6).shift(LEFT*4)
        mid  = NumberPlane().scale(0.6)
        right= NumberPlane().scale(0.6).shift(RIGHT*4)

        self.add(left, mid, right)

        t1 = Text("Indicator set (z-plane)").scale(0.45).next_to(left, UP)
        t2 = Text("Intermediate (ζ=e^{2πz})").scale(0.45).next_to(mid, UP)
        t3 = Text("Leaf (w=(ζ+1)/(ζ-1))").scale(0.45).next_to(right, UP)
        self.add(t1, t2, t3)

        # Example indicator: horizontal line y = omega
        omega = 0.25
        xs = np.linspace(-2, 2, 900)
        pts_z = np.stack([xs, omega*np.ones_like(xs), np.zeros_like(xs)], axis=1)

        ind = VMobject().set_points_smoothly(pts_z).set_stroke(width=6)
        ind.shift(LEFT*4)

        # Map to intermediate: point cloud for stability
        pts_mid = self._map_panel(pts_z, kind="exp")          # ζ-plane points
        pts_leaf = self._map_panel(pts_z, kind="leaf")        # w-plane points

        mid_dots = point_cloud(pts_mid, radius=0.018)
        right_dots = point_cloud(pts_leaf, radius=0.018)

        # place into panels
        mid_dots.shift(ORIGIN)
        right_dots.shift(RIGHT*4)

        # arrows
        a1 = Arrow(LEFT*2.3, LEFT*0.7)
        a2 = Arrow(RIGHT*0.7, RIGHT*2.3)

        self.play(ShowCreation(ind))
        self.play(GrowArrow(a1), FadeIn(mid_dots))
        self.play(GrowArrow(a2), FadeIn(right_dots))
        self.wait(1)

    def _map_panel(self, pts_z, kind="leaf"):
        # pts_z in (x,y,0) representing z=x+iy
        x = pts_z[:, 0]
        y = pts_z[:, 1]
        z = x + 1j*y
        if kind == "exp":
            zeta = exp_map(z)
            pts = np.stack([zeta.real, zeta.imag, np.zeros_like(zeta.real)], axis=1)
            good = np.isfinite(pts).all(axis=1)
            return pts[good] * 0.15  # scale into panel
        if kind == "leaf":
            zeta = exp_map(z)
            w = mobius(zeta)
            pts = np.stack([w.real, w.imag, np.zeros_like(w.real)], axis=1)
            good = np.isfinite(pts).all(axis=1) & (np.abs(w - 1) > 1e-6)
            return pts[good] * 1.2
        raise ValueError(kind)

# -------------------------
# Fig. 19–23 (indicator families → leaves)
# -------------------------

class Fig19_23_IndicatorFamiliesToLeaves(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        title = Text("Indicator families mapped to leaves (approx)").scale(0.55).to_corner(UL)
        self.add(title)

        # Animate widening strip in indicator plane, mapped to leaf point cloud
        xs = np.linspace(-2, 2, 900)
        omega0 = 0.15
        widths = np.linspace(0.0, 0.8, 60)

        cloud = VGroup()

        for w in widths:
            # strip: omega in [omega0-w/2, omega0+w/2]
            m = 12  # number of horizontal lines sampling the strip
            omegas = np.linspace(omega0 - w/2, omega0 + w/2, m)
            pts = []
            for om in omegas:
                y = om*np.ones_like(xs)
                pts.append(np.stack([xs, y], axis=1))
            pts = np.vstack(pts)
            pts3 = np.column_stack([pts, np.zeros(len(pts))])

            leaf_pts = map_indicator_to_leaf(np.column_stack([pts, np.zeros(len(pts))]))
            # replace cloud
            new_cloud = point_cloud(leaf_pts[::10], radius=0.015)  # decimate
            new_cloud.set_color(YELLOW)

            if len(cloud) == 0:
                cloud = new_cloud
                self.play(FadeIn(cloud))
            else:
                self.play(Transform(cloud, new_cloud), run_time=0.05)

        self.wait(1)

# -------------------------
# Fig. 24–25 (halo): add curved graphs alpha/beta in indicator, map → halo leaf
# -------------------------

class Fig24_25_LogLeafWithHalo(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        title = Text("Log leaf with halo (approx via α(x), β(x))").scale(0.55).to_corner(UL)
        self.add(title)

        xs = np.linspace(-2.0, 2.0, 600)
        base_omega = 0.15

        amps = np.linspace(0.0, 0.6, 60)

        cloud = VGroup()

        for a in amps:
            # alpha/beta as smooth bulges around base_omega
            alpha = base_omega + 0.15 + a*np.tanh(xs)
            beta  = base_omega - 0.15 - a*np.tanh(xs)

            # sample the region between beta and alpha by a few layers
            layers = 18
            ys = np.linspace(0, 1, layers)

            pts = []
            for s in ys:
                y = (1-s)*beta + s*alpha
                pts.append(np.stack([xs, y], axis=1))
            pts = np.vstack(pts)
            leaf_pts = map_indicator_to_leaf(np.column_stack([pts, np.zeros(len(pts))]))

            new_cloud = point_cloud(leaf_pts[::10], radius=0.015).set_color(YELLOW)

            if len(cloud) == 0:
                cloud = new_cloud
                self.play(FadeIn(cloud))
            else:
                self.play(Transform(cloud, new_cloud), run_time=0.05)

        self.wait(1)