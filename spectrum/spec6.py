

from manimlib import *
import numpy as np

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