from manimlib import *

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propedeutico_manim.common.algebra_helpers import run_step_by_step_solution


class MiercolesBinomioPotencias(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: binomio x^n +/- y^n",
            exercise=R"a^5-b^5",
            objective="Generalizar diferencia de cubos\na potencias iguales.",
            steps=[
                R"a^5-b^5",
                R"=(a-b)(a^4+a^3b+a^2b^2+ab^3+b^4)",
                R"\text{Patron: los exponentes de }a\text{ bajan y los de }b\text{ suben}",
                R"(a-b)(a^4+a^3b+a^2b^2+ab^3+b^4)",
            ],
            notes=[
                "Toda diferencia de potencias\ncomparte el factor a-b.",
                "El segundo factor tiene\n5 terminos.",
                "Cada termino conserva\ngrado total 4.",
                "El producto completo\nes la factorizacion.",
            ],
            result=R"a^5-b^5=(a-b)(a^4+a^3b+a^2b^2+ab^3+b^4)",
            step_font_size=30,
            result_font_size=29,
        )


class MiercolesFraccionCompleja(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: fraccion compleja",
            exercise=R"\frac{x-2}{x+2-\frac{4}{x-1}}",
            objective="Eliminar denominadores internos\ny cancelar solo factores.",
            steps=[
                R"\frac{x-2}{x+2-\frac{4}{x-1}}",
                R"=\frac{(x-1)(x-2)}{(x-1)(x+2)-4}",
                R"=\frac{(x-1)(x-2)}{x^2+x-6}",
                R"=\frac{(x-1)(x-2)}{(x+3)(x-2)}",
                R"=\frac{x-1}{x+3}",
            ],
            notes=[
                "La raya principal indica\nuna division completa.",
                "Multiplicamos arriba y abajo\npor x-1.",
                "Desarrollamos el denominador\nprincipal.",
                "Factorizamos antes\nde cancelar.",
                "Se cancela el factor x-2,\nno terminos sueltos.",
            ],
            result=R"\frac{x-2}{x+2-\frac{4}{x-1}}=\frac{x-1}{x+3}",
            step_font_size=30,
            result_font_size=31,
        )


class MiercolesSumaPotenciasImpares(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: suma de potencias impares",
            exercise=R"x^7+128",
            objective="Aplicar el patron de suma\ncuando el exponente es impar.",
            steps=[
                R"x^7+128",
                R"=x^7+2^7",
                R"=(x+2)(x^6-2x^5+4x^4-8x^3+16x^2-32x+64)",
                R"\text{Los signos alternan en el segundo factor}",
            ],
            notes=[
                "128 es 2^7.",
                "Como 7 es impar,\nx+2 es factor.",
                "Las potencias bajan en x\ny suben en 2.",
                "En suma impar,\nlos signos alternan.",
            ],
            result=R"x^7+128=(x+2)(x^6-2x^5+4x^4-8x^3+16x^2-32x+64)",
            step_font_size=27,
            result_font_size=25,
        )


class MiercolesVariosFactores(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: varios factores",
            exercise=R"a^6-b^6",
            objective="Seguir factorizando hasta\nseparar varios patrones conocidos.",
            steps=[
                R"a^6-b^6",
                R"=(a^3-b^3)(a^3+b^3)",
                R"=(a-b)(a^2+ab+b^2)(a+b)(a^2-ab+b^2)",
                R"=(a-b)(a+b)(a^2+ab+b^2)(a^2-ab+b^2)",
            ],
            notes=[
                "Primero vemos diferencia\nde cuadrados: (a^3)^2-(b^3)^2.",
                "Quedan una diferencia\ny una suma de cubos.",
                "Factorizamos cada cubo\ncon su formula.",
                "Reordenamos los factores\npara leerlos mejor.",
            ],
            result=R"a^6-b^6=(a-b)(a+b)(a^2+ab+b^2)(a^2-ab+b^2)",
            step_font_size=29,
            result_font_size=27,
        )


class MiercolesMetodoEvaluacion(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: metodo de evaluacion",
            exercise=R"x^3+2x^2-x-2",
            objective="Probar una raiz para obtener\nun factor lineal.",
            steps=[
                R"P(x)=x^3+2x^2-x-2",
                R"P(1)=1+2-1-2=0",
                R"\text{Entonces }x-1\text{ es factor}",
                R"x^3+2x^2-x-2=(x-1)(x^2+3x+2)",
                R"=(x-1)(x+1)(x+2)",
            ],
            notes=[
                "Nombramos el polinomio\npara evaluarlo.",
                "Al sustituir x=1\nobtenemos cero.",
                "Si P(r)=0, entonces\nx-r es factor.",
                "Dividimos entre x-1\ny obtenemos un trinomio.",
                "El trinomio restante\nse factoriza facil.",
            ],
            result=R"x^3+2x^2-x-2=(x-1)(x+1)(x+2)",
            step_font_size=30,
            result_font_size=30,
        )


class MiercolesIndeterminacionAparente(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Miercoles: indeterminacion aparente",
            exercise=R"\left.\frac{x^2-4}{x^2+x-6}\right|_{x=2}",
            objective="Simplificar antes de evaluar\ncuando aparece 0/0.",
            steps=[
                R"\frac{x^2-4}{x^2+x-6}",
                R"=\frac{(x-2)(x+2)}{(x-2)(x+3)}",
                R"=\frac{x+2}{x+3},\qquad x\neq2",
                R"\left.\frac{x+2}{x+3}\right|_{x=2}=\frac{4}{5}",
            ],
            notes=[
                "Sustituir directo da 0/0.",
                "Factorizamos numerador\ny denominador.",
                "Cancelamos el factor comun\npara hallar la expresion reducida.",
                "Ahora evaluamos en x=2.",
            ],
            result=R"\left.\frac{x^2-4}{x^2+x-6}\right|_{x=2}=\frac45",
            step_font_size=31,
            result_font_size=31,
        )
