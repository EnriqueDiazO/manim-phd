from manimlib import *

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propedeutico_manim.common.algebra_helpers import (
    equation_column,
    exercise_block,
    fit_to_width,
    objective_block,
    result_box as make_result_box,
    run_step_by_step_solution,
    scene_title,
    text_panel,
)


class LunesFactorComunSimple(Scene):
    def construct(self):
        title = scene_title("Lunes: factor comun simple")
        exercise_card = exercise_block(
            R"18mxy^2-54m^2x^2y^2+36my^2",
            width=6.4,
            font_size=25,
        )
        objective_card = objective_block(
            "Extraer el mayor factor comun\ny escribir el producto equivalente.",
            width=5.6,
        )

        cards_group = VGroup(exercise_card, objective_card)
        cards_group.arrange(RIGHT, aligned_edge=UP, buff=0.4)
        cards_group.next_to(title, DOWN, buff=0.28)
        fit_to_width(cards_group, 12.8)
        top_group = VGroup(title, cards_group)

        key_box = text_panel(
            "Paso clave",
            "Todos los terminos comparten\n18, m y y^2. Primero se identifica\nese factor; despues se divide\ncada termino entre el.",
            width=7.2,
            font_size=20,
            stroke_color=YELLOW,
        )
        key_box.next_to(cards_group, DOWN, buff=0.48)

        equations_group = equation_column(
            [
                R"18mxy^2-54m^2x^2y^2+36my^2",
                R"=18my^2(x)-18my^2(3mx^2)+18my^2(2)",
                R"=18my^2(x-3mx^2+2)",
            ],
            font_size=31,
            max_width=11.2,
            buff=0.38,
        )
        equations_group.next_to(top_group, DOWN, buff=0.58)
        equations_group.to_edge(LEFT, buff=0.9)

        verification = Tex(
            R"\text{Verificacion: }18my^2\cdot x-18my^2\cdot 3mx^2+18my^2\cdot 2",
            font_size=26,
        )
        fit_to_width(verification, 11.6)
        verification.next_to(equations_group, DOWN, buff=0.45)
        verification.align_to(equations_group, LEFT)

        result_box = make_result_box(
            R"18mxy^2-54m^2x^2y^2+36my^2=18my^2(x-3mx^2+2)",
            font_size=30,
        )
        result_box.to_edge(DOWN, buff=0.35)

        self.play(Write(title))
        self.play(FadeIn(exercise_card), FadeIn(objective_card))
        self.play(FadeIn(key_box))
        self.wait(0.8)
        self.play(FadeOut(key_box))

        self.play(Write(equations_group[0]))
        for index in range(1, len(equations_group)):
            self.play(Write(equations_group[index]))
            self.wait(0.2)

        self.play(FadeIn(verification, shift=0.12 * UP))
        self.wait(0.5)
        self.play(FadeOut(verification))
        self.play(FadeIn(result_box, shift=0.15 * UP))
        self.wait(1.0)


class LunesAgrupacion(Scene):
    def construct(self):
        title = scene_title("Lunes: factor comun por agrupacion")
        exercise_card = exercise_block(R"ax+bx+ay+by", width=5.7, font_size=28)
        objective_card = objective_block(
            "Agrupar terminos para hacer\naparecer un factor comun.",
            width=5.7,
        )

        cards_group = VGroup(exercise_card, objective_card)
        cards_group.arrange(RIGHT, aligned_edge=UP, buff=0.45)
        cards_group.next_to(title, DOWN, buff=0.28)
        fit_to_width(cards_group, 12.4)
        top_group = VGroup(title, cards_group)

        key_box = text_panel(
            "Paso clave",
            "No hay un factor comun visible\nen los cuatro terminos. Agrupamos\npara hacer aparecer el mismo\nbinomio en dos bloques.",
            width=7.1,
            font_size=20,
            stroke_color=YELLOW,
        )
        key_box.next_to(cards_group, DOWN, buff=0.48)

        equations_group = equation_column(
            [
                R"ax+bx+ay+by",
                R"=(ax+bx)+(ay+by)",
                R"=x(a+b)+y(a+b)",
                R"=(a+b)(x+y)",
            ],
            font_size=34,
            max_width=9.5,
            buff=0.32,
        )
        equations_group.next_to(top_group, DOWN, buff=0.55)
        equations_group.to_edge(LEFT, buff=1.35)

        verification = Tex(
            R"\text{Verificacion: }(a+b)(x+y)=ax+ay+bx+by",
            font_size=27,
        )
        fit_to_width(verification, 10.5)
        verification.next_to(equations_group, DOWN, buff=0.4)
        verification.align_to(equations_group, LEFT)

        result_box = make_result_box(R"ax+bx+ay+by=(a+b)(x+y)", font_size=33)
        result_box.to_edge(DOWN, buff=0.35)

        self.play(Write(title))
        self.play(FadeIn(exercise_card), FadeIn(objective_card))
        self.play(FadeIn(key_box))
        self.wait(0.8)
        self.play(FadeOut(key_box))

        self.play(Write(equations_group[0]))
        for index in range(1, len(equations_group)):
            self.play(Write(equations_group[index]))
            self.wait(0.2)

        self.play(FadeIn(verification, shift=0.12 * UP))
        self.wait(0.5)
        self.play(FadeOut(verification))
        self.play(FadeIn(result_box, shift=0.15 * UP))
        self.wait(1.0)


class LunesDiferenciaCubos(Scene):
    def construct(self):
        title = scene_title("Lunes: diferencia de cubos")
        exercise_card = exercise_block(R"a^3-8", width=5.5, font_size=30)
        objective_card = objective_block(
            "Reconocer dos cubos perfectos\ny aplicar el patron correcto.",
            width=5.7,
        )

        cards_group = VGroup(exercise_card, objective_card)
        cards_group.arrange(RIGHT, aligned_edge=UP, buff=0.45)
        cards_group.next_to(title, DOWN, buff=0.28)
        fit_to_width(cards_group, 12.3)
        top_group = VGroup(title, cards_group)

        key_box = text_panel(
            "Paso clave",
            "La expresion es una resta de dos\ncubos: a^3 y 2^3. Se usa\nu^3-v^3=(u-v)(u^2+uv+v^2).",
            width=7.0,
            font_size=20,
            stroke_color=YELLOW,
        )
        key_box.next_to(cards_group, DOWN, buff=0.48)

        equations_group = equation_column(
            [
                R"a^3-8",
                R"=a^3-2^3",
                R"=(a-2)(a^2+a\cdot 2+2^2)",
                R"=(a-2)(a^2+2a+4)",
            ],
            font_size=34,
            max_width=9.8,
            buff=0.32,
        )
        equations_group.next_to(top_group, DOWN, buff=0.55)
        equations_group.to_edge(LEFT, buff=1.3)

        verification = Tex(
            R"\text{Verificacion: }(a-2)(a^2+2a+4)=a^3-8",
            font_size=27,
        )
        fit_to_width(verification, 10.5)
        verification.next_to(equations_group, DOWN, buff=0.4)
        verification.align_to(equations_group, LEFT)

        result_box = make_result_box(R"a^3-8=(a-2)(a^2+2a+4)", font_size=33)
        result_box.to_edge(DOWN, buff=0.35)

        self.play(Write(title))
        self.play(FadeIn(exercise_card), FadeIn(objective_card))
        self.play(FadeIn(key_box))
        self.wait(0.8)
        self.play(FadeOut(key_box))

        self.play(Write(equations_group[0]))
        for index in range(1, len(equations_group)):
            self.play(Write(equations_group[index]))
            self.wait(0.2)

        self.play(FadeIn(verification, shift=0.12 * UP))
        self.wait(0.5)
        self.play(FadeOut(verification))
        self.play(FadeIn(result_box, shift=0.15 * UP))
        self.wait(1.0)


class LunesFactorComunPolinomio(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Lunes: factor comun polinomio",
            exercise=R"2x(a-1)-y(a-1)",
            objective="Detectar un binomio repetido\ny tratarlo como factor comun.",
            steps=[
                R"2x(a-1)-y(a-1)",
                R"=(a-1)(2x)-(a-1)(y)",
                R"=(a-1)(2x-y)",
                R"\text{Verificacion: }(a-1)(2x-y)=2x(a-1)-y(a-1)",
            ],
            notes=[
                "El factor repetido es\nel binomio a-1.",
                "Reescribimos cada termino\ncon el mismo factor adelante.",
                "Sacamos a-1 y dejamos\nlos cocientes 2x y -y.",
                "Distribuir confirma que\nno cambiamos la expresion.",
            ],
            result=R"2x(a-1)-y(a-1)=(a-1)(2x-y)",
            step_font_size=31,
            result_font_size=31,
        )


class LunesAgrupacionConSignos(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Lunes: agrupacion con signos",
            exercise=R"2x^2-3xy-4x+6y",
            objective="Agrupar cuidando el signo\nque queda fuera del segundo bloque.",
            steps=[
                R"2x^2-3xy-4x+6y",
                R"=(2x^2-3xy)-(4x-6y)",
                R"=x(2x-3y)-2(2x-3y)",
                R"=(2x-3y)(x-2)",
            ],
            notes=[
                "Agrupamos para buscar\nun binomio repetido.",
                "El signo menos obliga a\ncambiar signos dentro del bloque.",
                "Los dos grupos contienen\n2x-3y.",
                "Extraemos el binomio comun\ny quedan x y -2.",
            ],
            result=R"2x^2-3xy-4x+6y=(2x-3y)(x-2)",
            step_font_size=31,
            result_font_size=31,
        )


class LunesCuboPerfectoBinomio(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Lunes: cubo perfecto de binomio",
            exercise=R"8x^3+12x^2+6x+1",
            objective="Verificar el patron de\nun cubo de binomio.",
            steps=[
                R"8x^3+12x^2+6x+1",
                R"=(2x)^3+3(2x)^2(1)+3(2x)(1)^2+1^3",
                R"=(2x+1)^3",
                R"\text{Porque }(a+b)^3=a^3+3a^2b+3ab^2+b^3",
            ],
            notes=[
                "El primer termino es\n(2x)^3 y el ultimo es 1^3.",
                "Los terminos centrales\ncoinciden con los triples.",
                "Entonces la raiz del cubo\nes el binomio 2x+1.",
                "La formula explica\ncada termino del desarrollo.",
            ],
            result=R"8x^3+12x^2+6x+1=(2x+1)^3",
            step_font_size=30,
            result_font_size=31,
        )


class LunesProblemaProductoSuma(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Lunes: problema por factorizacion",
            exercise=(
                "Encontrar enteros positivos x,y\n"
                "cuyo producto sea 5 veces su suma."
            ),
            exercise_is_text=True,
            objective="Completar un producto para\nreducir el problema a factores.",
            steps=[
                R"xy=5(x+y)",
                R"xy-5x-5y=0",
                R"xy-5x-5y+25=25",
                R"(x-5)(y-5)=25",
                R"(x-5,y-5)\in\{(1,25),(5,5),(25,1)\}",
                R"(x,y)\in\{(6,30),(10,10),(30,6)\}",
            ],
            notes=[
                "Traducimos la condicion\ndel enunciado.",
                "Pasamos todo salvo el cero\nal mismo miembro.",
                "Sumamos 25 para completar\nla factorizacion.",
                "Ahora aparece un producto\nigual a 25.",
                "Como son enteros positivos,\nprobamos pares de factores positivos.",
                "Regresamos de x-5,y-5\na x,y.",
            ],
            result=R"(x,y)=(6,30),(10,10),(30,6)",
            step_font_size=28,
            result_font_size=31,
        )
