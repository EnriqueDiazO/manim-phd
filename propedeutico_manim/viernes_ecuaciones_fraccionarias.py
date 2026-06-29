from manimlib import *

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propedeutico_manim.common.algebra_helpers import run_step_by_step_solution


class ViernesEcuacionFraccionaria(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: ecuacion fraccionaria",
            exercise=R"\frac{3x}{5}-\frac{2x}{3}+\frac15=0",
            objective="Usar el mcm para eliminar\ndenominadores numericos.",
            steps=[
                R"\frac{3x}{5}-\frac{2x}{3}+\frac15=0",
                R"\operatorname{mcm}(5,3)=15",
                R"15\left(\frac{3x}{5}-\frac{2x}{3}+\frac15\right)=15\cdot0",
                R"9x-10x+3=0",
                R"-x+3=0",
                R"x=3",
            ],
            notes=[
                "Los denominadores son\n5 y 3.",
                "El mcm comun es 15.",
                "Multiplicamos toda\nla ecuacion por 15.",
                "Cada fraccion pierde\nsu denominador.",
                "Reducimos terminos semejantes.",
                "Despejamos x.",
            ],
            result=R"x=3",
            step_font_size=29,
        )


class ViernesProblemaFracciones(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: problema con fracciones",
            exercise=(
                "Despues de gastar la mitad\n"
                "y prestar la mitad de lo que quedo,\n"
                "tengo 21 quetzales."
            ),
            exercise_is_text=True,
            objective="Elegir la cantidad inicial\ncomo incognita y seguir las partes.",
            steps=[
                R"\text{Cantidad inicial}=x",
                R"\text{Despues de gastar la mitad: }\frac{x}{2}",
                R"\text{Despues de prestar la mitad de lo que quedo: }\frac{x}{4}",
                R"\frac{x}{4}=21",
                R"x=84",
            ],
            notes=[
                "La cantidad inicial es\nlo que queremos encontrar.",
                "Gastar la mitad deja\nla otra mitad.",
                "Prestar la mitad de x/2\ndeja x/4.",
                "Lo que queda son\n21 quetzales.",
                "Multiplicamos por 4.",
            ],
            result=R"\text{Cantidad inicial}=84\text{ quetzales}",
            step_font_size=29,
            result_font_size=32,
        )


class ViernesEcuacionDenominadoresAlgebraicos(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: denominadores algebraicos",
            exercise=R"\frac{5x+8}{3x+4}=\frac{5x+2}{3x-4}",
            objective="Anotar restricciones y\nmultiplicar por el mcm algebraico.",
            steps=[
                R"\frac{5x+8}{3x+4}=\frac{5x+2}{3x-4}",
                R"x\neq-\frac43,\qquad x\neq\frac43",
                R"(5x+8)(3x-4)=(5x+2)(3x+4)",
                R"15x^2+4x-32=15x^2+26x+8",
                R"-22x=40",
                R"x=-\frac{20}{11}",
            ],
            notes=[
                "Primero prohibimos los\nvalores que anulan denominadores.",
                "El mcm es\n(3x+4)(3x-4).",
                "Multiplicamos toda\nla ecuacion por el mcm.",
                "Desarrollamos ambos lados.",
                "Se cancelan los terminos\n15x^2.",
                "El resultado no viola\nlas restricciones.",
            ],
            result=R"x=-\frac{20}{11}",
            step_font_size=28,
            result_font_size=33,
        )


class ViernesEcuacionLiteralFraccionaria(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: ecuacion literal fraccionaria",
            exercise=R"\frac{m}{x}-\frac{1}{m}=\frac{2}{m}",
            objective="Eliminar denominadores y\nmantener las condiciones.",
            steps=[
                R"\frac{m}{x}-\frac{1}{m}=\frac{2}{m}",
                R"m\neq0,\qquad x\neq0",
                R"mx\left(\frac{m}{x}-\frac{1}{m}\right)=mx\left(\frac{2}{m}\right)",
                R"m^2-x=2x",
                R"m^2=3x",
                R"x=\frac{m^2}{3}",
            ],
            notes=[
                "Las letras tambien pueden\nestar en restricciones.",
                "Multiplicamos por mx\npara quitar denominadores.",
                "Cada termino recibe\nel mismo factor mx.",
                "Queda una ecuacion entera.",
                "Reunimos los terminos con x.",
                "Dividimos entre 3.",
            ],
            result=R"x=\frac{m^2}{3},\qquad m\neq0",
            step_font_size=28,
            result_font_size=32,
        )


class ViernesNumeroDisminuido(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: numero disminuido",
            exercise=(
                "Hallar el numero que disminuido\n"
                "en sus 3/8 equivale a su doble\n"
                "disminuido en 11."
            ),
            exercise_is_text=True,
            objective="Convertir el enunciado\na una ecuacion con fracciones.",
            steps=[
                R"\text{Numero}=x",
                R"x-\frac{3}{8}x=2x-11",
                R"\frac{5}{8}x=2x-11",
                R"5x=16x-88",
                R"11x=88",
                R"x=8",
            ],
            notes=[
                "La incognita es el numero.",
                "Disminuido en sus 3/8\nsignifica restar 3x/8.",
                "Reducimos el lado izquierdo.",
                "Multiplicamos por 8.",
                "Reunimos las x.",
                "El numero buscado es 8.",
            ],
            result=R"\text{El numero es }8",
            step_font_size=30,
            result_font_size=33,
        )


class ViernesNumerosConsecutivosFracciones(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: consecutivos y fracciones",
            exercise=(
                "Hallar dos numeros consecutivos\n"
                "tales que los 4/5 del mayor\n"
                "equivalgan al menor disminuido en 4."
            ),
            exercise_is_text=True,
            objective="Representar consecutivos\ny resolver la ecuacion fraccionaria.",
            steps=[
                R"\text{Menor}=x,\qquad \text{Mayor}=x+1",
                R"\frac45(x+1)=x-4",
                R"4(x+1)=5(x-4)",
                R"4x+4=5x-20",
                R"x=24",
                R"\text{Numeros: }24\text{ y }25",
            ],
            notes=[
                "Los consecutivos se escriben\ncomo x y x+1.",
                "La condicion compara\n4/5 del mayor con x-4.",
                "Multiplicamos por 5.",
                "Desarrollamos ambos lados.",
                "El menor es 24.",
                "El mayor es 25.",
            ],
            result=R"\text{Los numeros son }24\text{ y }25",
            step_font_size=29,
            result_font_size=32,
        )


class ViernesProblemaFraccionNumerador(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Viernes: problema con una fraccion",
            exercise=(
                "El numerador excede al denominador en 2.\n"
                "Si el denominador aumenta en 7,\n"
                "el valor de la fraccion es 1/2."
            ),
            exercise_is_text=True,
            objective="Traducir numerador y denominador\na expresiones con una incognita.",
            steps=[
                R"\text{Denominador}=x,\qquad \text{Numerador}=x+2",
                R"\frac{x+2}{x+7}=\frac12",
                R"2(x+2)=x+7",
                R"2x+4=x+7",
                R"x=3",
                R"\frac{x+2}{x}=\frac{5}{3}",
            ],
            notes=[
                "Elegimos x como denominador\nporque el numerador excede por 2.",
                "El denominador nuevo es x+7.",
                "Multiplicamos en cruz.",
                "Desarrollamos.",
                "El denominador original es 3.",
                "La fraccion original es 5/3.",
            ],
            result=R"\text{La fraccion original es }\frac{5}{3}",
            step_font_size=29,
            result_font_size=32,
        )
