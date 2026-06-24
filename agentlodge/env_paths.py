"""Resolve companion virtualenv paths for Lodge++ and EDGE."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


def resolve_venv_site_packages(code_path: Path) -> Path | None:
    venv_lib = code_path / ".venv" / "lib"
    if not venv_lib.exists():
        return None
    matches = sorted(venv_lib.glob("python*/site-packages"))
    return matches[0] if matches else None


def resolve_venv_python(code_path: Path) -> Path | None:
    venv_python = code_path / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else None


@contextmanager
def use_code_paths(*paths: Path | None):
    added: list[str] = []
    for path in paths:
        if path is None:
            continue
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added.append(text)
    try:
        yield
    finally:
        for text in added:
            if text in sys.path:
                sys.path.remove(text)


def edge_import_paths(edge_code_path: Path) -> list[Path]:
    paths = [edge_code_path]
    site = resolve_venv_site_packages(edge_code_path)
    if site is not None:
        paths.append(site)
    return paths


def lodge_import_paths(lodge_code_path: Path) -> list[Path]:
    paths = [lodge_code_path]
    site = resolve_venv_site_packages(lodge_code_path)
    if site is not None:
        paths.append(site)
    return paths
