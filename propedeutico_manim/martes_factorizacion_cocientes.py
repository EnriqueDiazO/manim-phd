from manimlib import *

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propedeutico_manim.common.algebra_helpers import run_step_by_step_solution


class MartesTrinomioCuadradoPorAdicion(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: adicion y sustraccion",
            exercise=R"x^4+x^2y^2+y^4",
            objective="Completar un cuadrado sin\ncambiar el valor del polinomio.",
            steps=[
                R"x^4+x^2y^2+y^4",
                R"=x^4+2x^2y^2+y^4-x^2y^2",
                R"=(x^2+y^2)^2-(xy)^2",
                R"=(x^2+y^2+xy)(x^2+y^2-xy)",
            ],
            notes=[
                "Los extremos son cuadrados:\nx^4 y y^4.",
                "Sumamos y restamos x^2y^2\npara formar el cuadrado.",
                "Queda una diferencia\nde cuadrados.",
                "Aplicamos U^2-V^2=(U+V)(U-V).",
            ],
            result=R"x^4+x^2y^2+y^4=(x^2+xy+y^2)(x^2-xy+y^2)",
            step_font_size=31,
            result_font_size=30,
        )


class MartesTrinomioBasico(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: trinomio x^2+bx+c",
            exercise=R"x^2+7x+10",
            objective="Buscar dos numeros cuya suma\nsea b y cuyo producto sea c.",
            steps=[
                R"x^2+7x+10",
                R"r+s=7,\qquad rs=10",
                R"5+2=7,\qquad 5\cdot2=10",
                R"x^2+7x+10=(x+5)(x+2)",
            ],
            notes=[
                "El trinomio es monico:\nel coeficiente de x^2 es 1.",
                "La suma debe dar 7\ny el producto debe dar 10.",
                "La pareja que funciona\nes 5 y 2.",
                "Escribimos un binomio\npor cada numero encontrado.",
            ],
            result=R"x^2+7x+10=(x+5)(x+2)",
        )


class MartesCocienteNotableBasico(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: cociente notable basico",
            exercise=R"\frac{9x^2-y^2}{3x+y}",
            objective="Leer el cociente como inversa\nde una diferencia de cuadrados.",
            steps=[
                R"\frac{9x^2-y^2}{3x+y}",
                R"=\frac{(3x)^2-y^2}{3x+y}",
                R"=\frac{(3x+y)(3x-y)}{3x+y}",
                R"=3x-y,\qquad 3x+y\neq 0",
            ],
            notes=[
                "El numerador es una\ndiferencia de cuadrados.",
                "Reconocemos a=3x\ny b=y.",
                "Factorizamos a^2-b^2\ncomo (a+b)(a-b).",
                "Cancelamos solo factores,\ncon denominador no nulo.",
            ],
            result=R"\frac{9x^2-y^2}{3x+y}=3x-y,\qquad 3x+y\neq0",
            step_font_size=34,
            result_font_size=31,
        )


class MartesSophieGermain(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: suma especial de cuadrados",
            exercise=R"a^4+4b^4",
            objective="Agregar y quitar el termino\nque permite diferencia de cuadrados.",
            steps=[
                R"a^4+4b^4",
                R"=a^4+4a^2b^2+4b^4-4a^2b^2",
                R"=(a^2+2b^2)^2-(2ab)^2",
                R"=(a^2+2b^2+2ab)(a^2+2b^2-2ab)",
            ],
            notes=[
                "Una suma de cuadrados no\nse factoriza siempre.",
                "Agregamos y quitamos\n4a^2b^2.",
                "Lo que queda es una\ndiferencia de cuadrados.",
                "Aplicamos U^2-V^2.",
            ],
            result=R"a^4+4b^4=(a^2+2ab+2b^2)(a^2-2ab+2b^2)",
            step_font_size=30,
            result_font_size=29,
        )


class MartesTrinomioNoMonico(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: trinomio ax^2+bx+c",
            exercise=R"6x^2+7x+2",
            objective="Descomponer el termino medio\npara factorizar por agrupacion.",
            steps=[
                R"6x^2+7x+2",
                R"ac=6\cdot2=12,\qquad r+s=7,\quad rs=12",
                R"r=3,\qquad s=4",
                R"6x^2+7x+2=6x^2+3x+4x+2",
                R"=3x(2x+1)+2(2x+1)",
                R"=(3x+2)(2x+1)",
            ],
            notes=[
                "El coeficiente principal\nno es 1.",
                "Buscamos dos numeros\nque sumen 7 y multipliquen 12.",
                "La pareja es 3 y 4.",
                "Partimos 7x como 3x+4x.",
                "Ahora aparece el binomio\ncomun 2x+1.",
                "Extraemos el binomio comun.",
            ],
            result=R"6x^2+7x+2=(3x+2)(2x+1)",
            step_font_size=28,
            result_font_size=31,
        )


class MartesTrinomioConSignos(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: trinomio con signos",
            exercise=R"x^2+3x-10",
            objective="Usar suma y producto para\ndecidir los signos de los factores.",
            steps=[
                R"x^2+3x-10",
                R"r+s=3,\qquad rs=-10",
                R"5+(-2)=3,\qquad 5(-2)=-10",
                R"x^2+3x-10=(x+5)(x-2)",
            ],
            notes=[
                "Producto negativo: los signos\nson opuestos.",
                "La suma debe conservar\nel signo positivo de 3.",
                "El numero de mayor valor\nabsoluto es positivo.",
                "Escribimos los factores.",
            ],
            result=R"x^2+3x-10=(x+5)(x-2)",
        )


class MartesCocienteCubos(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Martes: cociente de cubos",
            exercise=R"\frac{8x^3+y^3}{2x+y}",
            objective="Reconocer suma de cubos\ny cancelar el factor lineal.",
            steps=[
                R"\frac{8x^3+y^3}{2x+y}",
                R"=\frac{(2x)^3+y^3}{2x+y}",
                R"=\frac{(2x+y)(4x^2-2xy+y^2)}{2x+y}",
                R"=4x^2-2xy+y^2,\qquad 2x+y\neq0",
            ],
            notes=[
                "8x^3 es el cubo de 2x.",
                "Usamos a^3+b^3=(a+b)(a^2-ab+b^2).",
                "El denominador coincide\ncon el factor lineal.",
                "Cancelamos factores,\nno terminos separados.",
            ],
            result=R"\frac{8x^3+y^3}{2x+y}=4x^2-2xy+y^2",
            step_font_size=31,
            result_font_size=31,
        )
