from manimlib import *

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from propedeutico_manim.common.algebra_helpers import run_step_by_step_solution


class JuevesEcuacionLineal(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: ecuacion lineal",
            exercise=R"8x+9-12x=4x-13-5x",
            objective="Reunir terminos semejantes,\ntransponer y despejar.",
            steps=[
                R"8x+9-12x=4x-13-5x",
                R"-4x+9=-x-13",
                R"-4x+x=-13-9",
                R"-3x=-22",
                R"x=\frac{22}{3}",
            ],
            notes=[
                "Primero reducimos terminos\nsemejantes en cada miembro.",
                "La izquierda queda -4x+9\ny la derecha -x-13.",
                "Pasamos las x a un lado\ny las constantes al otro.",
                "Reducimos ambos miembros.",
                "Dividimos toda la igualdad\nentre -3.",
            ],
            result=R"x=\frac{22}{3}",
        )


class JuevesProblemaVerbal(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: problema verbal",
            exercise=(
                "La suma de dos numeros es 106\n"
                "y el mayor excede al menor en 8."
            ),
            exercise_is_text=True,
            objective="Definir la incognita,\ntraducir el enunciado y resolver.",
            steps=[
                R"\text{Menor}=x,\qquad \text{Mayor}=x+8",
                R"x+(x+8)=106",
                R"2x+8=106",
                R"2x=98",
                R"x=49,\qquad x+8=57",
            ],
            notes=[
                "Elegimos x como el numero\nmenor para expresar el mayor.",
                "La palabra suma se traduce\ncon una igualdad.",
                "Quitamos parentesis\ny reducimos terminos.",
                "Restamos 8 en ambos lados.",
                "El menor es 49\ny el mayor es 57.",
            ],
            result=R"\text{Los numeros son }49\text{ y }57.",
            step_font_size=32,
            result_font_size=33,
        )


class JuevesEcuacionConParentesis(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: ecuacion con parentesis",
            exercise=R"5(x-1)+16(2x+3)=3(2x-7)-x",
            objective="Desarrollar productos,\nreducir y despejar.",
            steps=[
                R"5(x-1)+16(2x+3)=3(2x-7)-x",
                R"5x-5+32x+48=6x-21-x",
                R"37x+43=5x-21",
                R"32x=-64",
                R"x=-2",
            ],
            notes=[
                "Distribuimos cada numero\nsobre su parentesis.",
                "Cuidamos signos al quitar\nparentesis.",
                "Reducimos terminos semejantes\nen ambos miembros.",
                "Pasamos x a la izquierda\ny constantes a la derecha.",
                "Dividimos entre 32.",
            ],
            result=R"x=-2",
            step_font_size=29,
        )


class JuevesEdadesDobleTriple(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: problema de edades",
            exercise=(
                "A tiene el doble de edad que B.\n"
                "Hace 10 anos A tenia el triple que B."
            ),
            exercise_is_text=True,
            objective="Modelar edades actuales\ny edades de hace 10 anos.",
            steps=[
                R"\text{Edad de B}=x,\qquad \text{Edad de A}=2x",
                R"2x-10=3(x-10)",
                R"2x-10=3x-30",
                R"x=20",
                R"\text{B}=20,\qquad \text{A}=40",
            ],
            notes=[
                "Conviene elegir como x\nla edad de B.",
                "Hace 10 anos restamos 10\na cada edad.",
                "Desarrollamos el parentesis\nde la derecha.",
                "La edad de B es 20.",
                "La edad de A es el doble.",
            ],
            result=R"\text{B tiene }20\text{ anos y A tiene }40\text{ anos}",
            step_font_size=31,
            result_font_size=30,
        )


class JuevesPadreHijo(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: padre e hijo",
            exercise=(
                "Un padre tiene 35 anos y su hijo 5.\n"
                "En cuantos anos el padre tendra\n"
                "triple edad que el hijo?"
            ),
            exercise_is_text=True,
            objective="Usar la misma incognita\npara el tiempo que pasa.",
            steps=[
                R"\text{Anios que pasan}=t",
                R"35+t=3(5+t)",
                R"35+t=15+3t",
                R"20=2t",
                R"t=10",
            ],
            notes=[
                "La incognita es el tiempo\nque transcurre.",
                "Ambas edades aumentan\nen la misma cantidad t.",
                "La nueva edad del padre\nsera triple de la del hijo.",
                "Reducimos la ecuacion.",
                "Faltan 10 anos.",
            ],
            result=R"\text{Dentro de }10\text{ anos}",
            step_font_size=31,
            result_font_size=33,
        )


class JuevesEnjambre(Scene):
    def construct(self):
        run_step_by_step_solution(
            self,
            title="Jueves: problema del enjambre",
            exercise=(
                "La mitad queda en una flor,\n"
                "la tercera parte en otra\n"
                "y 5 siguen volando."
            ),
            exercise_is_text=True,
            objective="Traducir partes de un total\na una ecuacion lineal.",
            steps=[
                R"\text{Total}=x",
                R"\frac{x}{2}+\frac{x}{3}+5=x",
                R"3x+2x+30=6x",
                R"30=x",
                R"x=30",
            ],
            notes=[
                "El total del enjambre\nes la incognita.",
                "Las partes descritas\nsuman el total.",
                "Multiplicamos por 6\npara eliminar denominadores.",
                "Reducimos terminos.",
                "Hay 30 en total.",
            ],
            result=R"\text{Hay }30\text{ abejas}",
            step_font_size=31,
            result_font_size=33,
        )
