from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_VIDEO_DIR = "/tmp/manim_propedeutico_validate"
EXIT_WORDS = {"0", "q", "quit", "salir", "exit"}
BACK_WORDS = {"v", "volver", "back", "menu"}


@dataclass(frozen=True)
class SceneEntry:
    class_name: str
    file_path: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


SCENES_BY_DAY: dict[str, list[SceneEntry]] = {
    "Lunes": [
        SceneEntry("LunesFactorComunSimple", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesAgrupacion", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesDiferenciaCubos", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesFactorComunPolinomio", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesAgrupacionConSignos", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesCuboPerfectoBinomio", "propedeutico_manim/lunes_factorizacion.py"),
        SceneEntry("LunesProblemaProductoSuma", "propedeutico_manim/lunes_factorizacion.py"),
    ],
    "Martes": [
        SceneEntry("MartesTrinomioCuadradoPorAdicion", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesTrinomioBasico", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesCocienteNotableBasico", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesSophieGermain", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesTrinomioNoMonico", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesTrinomioConSignos", "propedeutico_manim/martes_factorizacion_cocientes.py"),
        SceneEntry("MartesCocienteCubos", "propedeutico_manim/martes_factorizacion_cocientes.py"),
    ],
    "Miercoles": [
        SceneEntry("MiercolesBinomioPotencias", "propedeutico_manim/miercoles_potencias_fracciones.py"),
        SceneEntry("MiercolesFraccionCompleja", "propedeutico_manim/miercoles_potencias_fracciones.py"),
        SceneEntry("MiercolesSumaPotenciasImpares", "propedeutico_manim/miercoles_potencias_fracciones.py"),
        SceneEntry("MiercolesVariosFactores", "propedeutico_manim/miercoles_potencias_fracciones.py"),
        SceneEntry("MiercolesMetodoEvaluacion", "propedeutico_manim/miercoles_potencias_fracciones.py"),
        SceneEntry("MiercolesIndeterminacionAparente", "propedeutico_manim/miercoles_potencias_fracciones.py"),
    ],
    "Jueves": [
        SceneEntry("JuevesEcuacionLineal", "propedeutico_manim/jueves_ecuaciones_lineales.py"),
        SceneEntry("JuevesProblemaVerbal", "propedeutico_manim/jueves_ecuaciones_lineales.py"),
        SceneEntry(
            "JuevesEcuacionConParentesis",
            "propedeutico_manim/jueves_ecuaciones_lineales.py",
            aliases=("JuevesEcuacionParentesis",),
        ),
        SceneEntry(
            "JuevesEdadesDobleTriple",
            "propedeutico_manim/jueves_ecuaciones_lineales.py",
            aliases=("JuevesProblemaEdades",),
        ),
        SceneEntry(
            "JuevesPadreHijo",
            "propedeutico_manim/jueves_ecuaciones_lineales.py",
            aliases=("JuevesProblemaPadreHijo",),
        ),
        SceneEntry(
            "JuevesEnjambre",
            "propedeutico_manim/jueves_ecuaciones_lineales.py",
            aliases=("JuevesProblemaEnjambre",),
        ),
    ],
    "Viernes": [
        SceneEntry("ViernesEcuacionFraccionaria", "propedeutico_manim/viernes_ecuaciones_fraccionarias.py"),
        SceneEntry("ViernesProblemaFracciones", "propedeutico_manim/viernes_ecuaciones_fraccionarias.py"),
        SceneEntry(
            "ViernesEcuacionDenominadoresAlgebraicos",
            "propedeutico_manim/viernes_ecuaciones_fraccionarias.py",
            aliases=("ViernesDenominadoresAlgebraicos",),
        ),
        SceneEntry("ViernesEcuacionLiteralFraccionaria", "propedeutico_manim/viernes_ecuaciones_fraccionarias.py"),
        SceneEntry("ViernesNumeroDisminuido", "propedeutico_manim/viernes_ecuaciones_fraccionarias.py"),
        SceneEntry(
            "ViernesNumerosConsecutivosFracciones",
            "propedeutico_manim/viernes_ecuaciones_fraccionarias.py",
            aliases=("ViernesConsecutivosFracciones",),
        ),
        SceneEntry(
            "ViernesProblemaFraccionNumerador",
            "propedeutico_manim/viernes_ecuaciones_fraccionarias.py",
            aliases=("ViernesProblemaNumeradorDenominador",),
        ),
    ],
}


def normalize_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.strip())
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return "".join(char for char in without_accents.casefold() if char.isalnum())


def is_exit(value: str) -> bool:
    return normalize_key(value) in EXIT_WORDS


def is_back(value: str) -> bool:
    return normalize_key(value) in BACK_WORDS


def get_scene_classes(file_path: str) -> set[str]:
    source_path = PROJECT_ROOT / file_path
    if not source_path.exists():
        return set()

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    classes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Scene":
                classes.add(node.name)
            elif isinstance(base, ast.Attribute) and base.attr == "Scene":
                classes.add(node.name)
    return classes


def validate_scenes() -> dict[str, set[str]]:
    files = {scene.file_path for scene in iter_scenes()}
    return {file_path: get_scene_classes(file_path) for file_path in files}


def iter_scenes() -> Iterable[SceneEntry]:
    for scenes in SCENES_BY_DAY.values():
        yield from scenes


def scene_exists(scene: SceneEntry, available: dict[str, set[str]]) -> bool:
    return scene.class_name in available.get(scene.file_path, set())


def aliases_for(scene: SceneEntry) -> tuple[str, ...]:
    auto_aliases = {
        scene.class_name.replace("Miercoles", "Miércoles"),
        scene.class_name.replace("Agrupacion", "Agrupación"),
        scene.class_name.replace("Ecuacion", "Ecuación"),
    }
    auto_aliases.discard(scene.class_name)
    return tuple(dict.fromkeys((*scene.aliases, *auto_aliases)))


def print_warnings(available: dict[str, set[str]]) -> None:
    missing_files = sorted(
        {scene.file_path for scene in iter_scenes() if not (PROJECT_ROOT / scene.file_path).exists()}
    )
    for file_path in missing_files:
        print(f"ADVERTENCIA: no existe el archivo declarado: {file_path}")

    for day, scenes in SCENES_BY_DAY.items():
        for scene in scenes:
            if not scene_exists(scene, available):
                print(
                    "ADVERTENCIA: "
                    f"{scene.class_name} esta declarada para {day}, "
                    f"pero no existe en {scene.file_path}."
                )


def print_scene_list(available: dict[str, set[str]]) -> None:
    print("Escenas del propedeutico:\n")
    for day, scenes in SCENES_BY_DAY.items():
        print(day)
        for scene in scenes:
            status = "OK" if scene_exists(scene, available) else "NO EXISTE"
            alias_text = ""
            scene_aliases = aliases_for(scene)
            if scene_aliases:
                alias_text = " | aliases: " + ", ".join(scene_aliases)
            print(f"  - {scene.class_name} [{status}] ({scene.file_path}){alias_text}")
        print()


def ask_run_mode() -> str | None:
    options = {"1": "interactive", "2": "static", "3": "video"}
    while True:
        print("Modo de ejecucion:")
        print("1. Interactivo")
        print("2. Estatico / validacion")
        print("3. Renderizar video")
        answer = input("Elige modo: ").strip()
        if is_exit(answer):
            return None
        if answer in options:
            return options[answer]
        normalized = normalize_key(answer)
        if normalized in {"interactivo", "interactive"}:
            return "interactive"
        if normalized in {"estatico", "validacion", "static"}:
            return "static"
        if normalized in {"video", "renderizarvideo"}:
            return "video"
        print("Opcion no valida. Escribe 1, 2, 3 o salir.\n")


def main_menu() -> str | None:
    day_names = list(SCENES_BY_DAY)
    while True:
        print("\nMenu principal:")
        for index, day in enumerate(day_names, start=1):
            print(f"{index}. {day}")
        print(f"{len(day_names) + 1}. Todas las escenas")
        print("0. Salir")

        answer = input("Elige dia: ").strip()
        if is_exit(answer):
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(day_names):
                return day_names[index - 1]
            if index == len(day_names) + 1:
                return "Todas"

        key = normalize_key(answer)
        day_lookup = {normalize_key(day): day for day in day_names}
        day_lookup.update(
            {
                "miercoles": "Miercoles",
                "todas": "Todas",
                "todo": "Todas",
                "all": "Todas",
                "todasescenas": "Todas",
            }
        )
        if key in day_lookup:
            return day_lookup[key]
        print("Opcion no valida. Elige un numero, un dia o salir.")


def scenes_for_choice(day_choice: str) -> list[tuple[str, SceneEntry]]:
    if day_choice == "Todas":
        return [(day, scene) for day, scenes in SCENES_BY_DAY.items() for scene in scenes]
    return [(day_choice, scene) for scene in SCENES_BY_DAY[day_choice]]


def choose_scene(day_choice: str, available: dict[str, set[str]]) -> SceneEntry | str | None:
    choices = scenes_for_choice(day_choice)
    while True:
        title = "Todas las escenas" if day_choice == "Todas" else day_choice
        print(f"\n{title}:")
        for index, (day, scene) in enumerate(choices, start=1):
            status = "" if scene_exists(scene, available) else " [NO EXISTE]"
            prefix = f"{day}: " if day_choice == "Todas" else ""
            print(f"{index}. {prefix}{scene.class_name}{status}")
        print("v. Volver al menu principal")
        print("0. Salir")

        answer = input("Elige escena por numero o nombre: ").strip()
        if is_exit(answer):
            return "EXIT"
        if is_back(answer):
            return None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(choices):
                return choices[index - 1][1]

        scene_lookup: dict[str, SceneEntry] = {}
        for _, scene in choices:
            scene_lookup[normalize_key(scene.class_name)] = scene
            for alias in aliases_for(scene):
                scene_lookup[normalize_key(alias)] = scene

        key = normalize_key(answer)
        if key in scene_lookup:
            return scene_lookup[key]
        print("Escena no encontrada en esta seleccion. Prueba numero, clase exacta o alias.")


def command_for(scene: SceneEntry, mode: str) -> list[str]:
    command = ["poetry", "run", "manimgl", scene.file_path, scene.class_name]
    if mode == "static":
        command.extend(["-s", "-w", "--video_dir", VALIDATION_VIDEO_DIR])
    elif mode == "video":
        command.append("-w")
    return command


def format_command(command: list[str]) -> str:
    return " ".join(command)


def run_scene(scene: SceneEntry, mode: str, available: dict[str, set[str]]) -> None:
    if not scene_exists(scene, available):
        print(
            "\nNo se puede ejecutar esta escena: "
            f"{scene.class_name} no existe en {scene.file_path}.\n"
        )
        return

    command = command_for(scene, mode)
    print("\nEjecutando:", flush=True)
    print(format_command(command), flush=True)
    print(flush=True)
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except KeyboardInterrupt:
        print("\nEjecucion interrumpida. Regresando al menu.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Menu interactivo para escenas del propedeutico en ManimGL."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista escenas declaradas, valida clases existentes y sale sin ejecutar Manim.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = validate_scenes()
    print_warnings(available)

    if args.list:
        print_scene_list(available)
        return 0

    mode = ask_run_mode()
    if mode is None:
        return 0

    while True:
        day_choice = main_menu()
        if day_choice is None:
            return 0

        scene = choose_scene(day_choice, available)
        if scene == "EXIT":
            return 0
        if scene is None:
            continue

        run_scene(scene, mode, available)


if __name__ == "__main__":
    sys.exit(main())
