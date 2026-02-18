from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(path: str = "requirements.txt") -> list[str]:
    reqs: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


setup(
    packages=find_packages(),
    install_requires=read_requirements(),
)
