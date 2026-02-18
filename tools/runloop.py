#!/usr/bin/env python3
"""
Run ManimGL scenes in a loop, selecting by number or name.

Usage:
  poetry run python tools/runloop.py path/to/file.py [-- extra manimgl args]

Examples:
  poetry run python tools/runloop.py spectrum/spectrumcopy.py
  poetry run python tools/runloop.py spectrum/spectrumcopy.py -- -w -o
  poetry run python tools/runloop.py spectrum/spectrumcopy.py -- -w -o -s
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional


SceneEntry = Tuple[str, type]


def load_scenes(py_file: Path) -> List[SceneEntry]:
    """
    Load a python file as a module and return (name, cls) for all Scene
    subclasses defined in that file (not imported bases).
    """
    from manimlib.scene.scene import Scene  # type: ignore

    module_name = f"_manim_loop_{py_file.stem}"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import from: {py_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore

    scenes: List[SceneEntry] = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, Scene) or obj is Scene:
            continue
        # Only classes defined in this file/module (not imported)
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        scenes.append((name, obj))

    scenes.sort(key=lambda t: t[0].lower())
    return scenes


def print_menu(py_file: Path, scenes: List[SceneEntry]) -> None:
    print("\n" + "-" * 72)
    print(f"File: {py_file}")
    print("Available scenes:")
    if not scenes:
        print("  (none found) -> No subclasses of Scene detected in this file.")
    else:
        for i, (name, _) in enumerate(scenes, start=1):
            print(f"  {i}: {name}")
    print("-" * 72)
    print("Commands: [number|name] run,  r reload,  f <file> switch file,  a autorun,  q quit")


def resolve_selection(sel: str, scenes: List[SceneEntry]) -> Optional[str]:
    sel = sel.strip()
    if not sel:
        return None

    # Number?
    if sel.isdigit():
        idx = int(sel)
        if 1 <= idx <= len(scenes):
            return scenes[idx - 1][0]
        return None

    # Name (exact)
    names = {name: name for name, _ in scenes}
    if sel in names:
        return sel

    # Name (case-insensitive)
    lower_map = {name.lower(): name for name, _ in scenes}
    return lower_map.get(sel.lower())


def run_scene(py_file: Path, scene_name: str, scenes: List[SceneEntry], extra_args: List[str]) -> None:
    idx = [name for name, _ in scenes].index(scene_name) + 1
    print(f"\n[RUN] #{idx}: {scene_name}\n")
    cmd = ["poetry", "run", "manimgl", str(py_file), scene_name, *extra_args]
    subprocess.run(cmd)


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        print("Usage: poetry run python tools/runloop.py path/to/file.py [-- extra manimgl args]")
        return 2

    # Split args: script args vs extra manimgl args after `--`
    if "--" in argv:
        sep = argv.index("--")
        file_arg = argv[0]
        extra_args = argv[sep + 1 :]
    else:
        file_arg = argv[0]
        extra_args = []

    py_file = Path(file_arg).expanduser().resolve()
    if not py_file.exists():
        print(f"Error: file not found: {py_file}")
        return 2

    while True:
        # Always reload scenes so menu reflects latest edits
        try:
            scenes = load_scenes(py_file)
        except Exception as e:
            print(f"\n[ERROR] Could not load scenes from {py_file}:\n  {e}")
            scenes = []

        print_menu(py_file, scenes)

        raw = input("Scene (number/name) > ").strip()
        if not raw:
            continue

        cmd = raw.strip()

        if cmd.lower() in {"q", "quit", "exit"}:
            break

        if cmd.lower() in {"r", "reload"}:
            continue

        if cmd.lower().startswith("f "):
            new_path = cmd[2:].strip()
            new_file = Path(new_path).expanduser().resolve()
            if not new_file.exists():
                print(f"[WARN] file not found: {new_file}")
            else:
                py_file = new_file
            continue

        if cmd.lower() in {"a", "auto"}:
            # autorun sequentially
            if not scenes:
                print("[WARN] No scenes to run.")
                continue
            for name, _ in scenes:
                run_scene(py_file, name, scenes, extra_args)
            continue

        scene_name = resolve_selection(cmd, scenes)
        if not scene_name:
            print(f"[WARN] Invalid selection: {cmd!r}")
            continue

        run_scene(py_file, scene_name, scenes, extra_args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
