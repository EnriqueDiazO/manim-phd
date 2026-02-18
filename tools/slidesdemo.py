# tools/slides_demo.py
from __future__ import annotations

from manimlib import *


class ClickOrKeyToAdvance(Scene):
    """
    Demo: avanza SOLO cuando llega un evento de teclado/click.
    Ejecuta:
      poetry run manimgl tools/slides_demo.py ClickOrKeyToAdvance
    """

    def setup(self):
        super().setup()
        self._advance = False
        self._installed = False

        win = getattr(self, "window", None)
        if win is None:
            return

        # ---- Intento 1: event_dispatcher (ManimGL >=1.6 suele tenerlo)
        dispatcher = getattr(win, "event_dispatcher", None)
        if dispatcher is not None:
            # Buscamos EventType dentro del repo de manimlib
            try:
                from manimlib.event_handler.event_type import EventType  # type: ignore
            except Exception:
                EventType = None  # noqa: N806

            if EventType is not None:
                # Compat: algunos nombres cambian, intentamos varias opciones
                def try_add(event_name: str, handler):
                    et = getattr(EventType, event_name, None)
                    if et is not None:
                        try:
                            dispatcher.add_listener(et, handler)
                            return True
                        except Exception:
                            return False
                    return False

                ok1 = try_add("KEY_PRESS", self._on_key_event)
                ok2 = try_add("MOUSE_PRESS", self._on_mouse_event) or try_add("MOUSE_BUTTON_PRESS", self._on_mouse_event)

                if ok1 or ok2:
                    self._installed = True
                    return

        # ---- Intento 2: APIs directas (algunas builds tienen métodos "add_*")
        for meth_name, handler in [
            ("add_key_press_listener", self._on_key_event),
            ("add_mouse_press_listener", self._on_mouse_event),
            ("add_key_listener", self._on_key_event),
            ("add_mouse_listener", self._on_mouse_event),
        ]:
            meth = getattr(win, meth_name, None)
            if callable(meth):
                try:
                    meth(handler)
                    self._installed = True
                except Exception:
                    pass

    # Los callbacks reciben distintos “shapes” según versión; los hacemos tolerantes.
    def _on_key_event(self, *args, **kwargs):
        """
        Dispara avance con SPACE o ENTER.
        En algunas versiones args trae (symbol, modifiers) o un event object.
        """
        # Caso: event object con .key / .symbol / .name
        if args:
            ev = args[0]
            for attr in ("key", "symbol", "name"):
                if hasattr(ev, attr):
                    val = getattr(ev, attr)
                    if str(val).lower() in {"space", "enter", "return"}:
                        self._advance = True
                        return

        # Caso: (symbol, modifiers) como enteros/códigos
        if len(args) >= 1:
            sym = args[0]
            # A veces llega string 'SPACE'
            if isinstance(sym, str) and sym.lower() in {"space", "enter", "return"}:
                self._advance = True
                return

        # Si no podemos reconocer, cualquier tecla avanza (opcional)
        self._advance = True

    def _on_mouse_event(self, *args, **kwargs):
        self._advance = True

    def wait_for_advance(self):
        """
        Espera hasta que ocurra un evento; si no se instaló input-hook,
        cae a "ENTER en terminal" (fallback).
        """
        self._advance = False

        if not self._installed:
            # fallback fiable
            print("[slides_demo] No pude enganchar eventos de ventana. Pulsa ENTER en terminal para avanzar...")
            input()
            return

        # ManimGL suele soportar stop_condition en wait()
        self.wait(1e9, stop_condition=lambda: self._advance)

    def construct(self):
        title = Text("Slide-show mode", font_size=56).to_edge(UP)
        hint = Text("Click o cualquier tecla para avanzar (ideal: SPACE/ENTER)", font_size=28).to_edge(DOWN)
        self.add(title, hint)

        dot = Dot(LEFT * 3)
        self.play(FadeIn(dot))
        self.wait_for_advance()

        self.play(dot.animate.shift(RIGHT * 6), run_time=1.0)
        self.wait_for_advance()

        circle = Circle().next_to(dot, UP)
        self.play(ShowCreation(circle))
        self.wait_for_advance()

        self.play(FadeOut(VGroup(dot, circle, title, hint)))



from manimlib import *
from pyglet.window import key as Keys
import numpy as np


class ButtonNextDemo(InteractiveScene):
    def setup(self):
        super().setup()
        self._advance = False
        self.next_button = self._make_button()

    def _make_button(self):
        box = RoundedRectangle(corner_radius=0.15, width=3.0, height=0.8)
        box.set_stroke(WHITE, 2).set_fill(BLACK, 0.2)
        label = Text("NEXT  ▶", font_size=28)
        btn = VGroup(box, label)
        btn.arrange(RIGHT, buff=0.2)
        btn.to_edge(DOWN).shift(UP * 0.2)
        btn.fix_in_frame()  # para que sea tipo UI
        self.add(btn)
        return btn

    def _mouse_in_button(self) -> bool:
        # mouse_point está en coords de escena; convertimos a fixed frame coords
        p_scene = self.mouse_point.get_center()
        p_fixed = self.frame.to_fixed_frame_point(p_scene)

        # bounding box del botón (también en fixed frame porque fix_in_frame)
        bb = self.next_button.get_bounding_box()
        mins = bb.min(axis=0)
        maxs = bb.max(axis=0)
        return bool(np.all(p_fixed >= mins) and np.all(p_fixed <= maxs))

    def on_key_press(self, symbol, modifiers):
        if symbol in (Keys.SPACE, Keys.ENTER, Keys.NUM_ENTER):
            self._advance = True
        return super().on_key_press(symbol, modifiers)

    def on_mouse_press(self, *args, **kwargs):
        if self._mouse_in_button():
            self._advance = True
        meth = getattr(super(), "on_mouse_press", None)
        if callable(meth):
            return meth(*args, **kwargs)

    def wait_for_advance(self):
        self._advance = False
        self.wait(1e9, stop_condition=lambda: self._advance)

    def construct(self):
        title = Text("Click NEXT or press SPACE", font_size=48).to_edge(UP)
        title.fix_in_frame()
        self.add(title)

        dot = Dot(LEFT * 3)
        self.play(FadeIn(dot))
        self.wait_for_advance()

        self.play(dot.animate.shift(RIGHT * 6), run_time=1.0)
        self.wait_for_advance()

        circle = Circle().next_to(dot, UP)
        self.play(ShowCreation(circle))
        self.wait_for_advance()

        self.play(FadeOut(VGroup(dot, circle)))
