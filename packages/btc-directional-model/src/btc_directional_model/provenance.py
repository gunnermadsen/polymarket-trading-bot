from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DEPENDENCIES = (
    "numpy",
    "plotly",
    "polars",
    "psycopg",
    "pyarrow",
    "scikit-learn",
    "scipy",
)


def runtime_provenance(package_root: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": {name: installed_version(name) for name in DEPENDENCIES},
        "source_tree_sha256": source_tree_sha256(package_root),
        "git": git_provenance(package_root.parent.parent),
    }


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def source_tree_sha256(package_root: Path) -> str:
    candidates = [
        package_root / "pyproject.toml",
        package_root / "requirements.lock",
        *sorted((package_root / "configs").glob("*.toml")),
        *sorted((package_root / "sql").glob("*.sql")),
        *sorted((package_root / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in candidates:
        if not path.exists():
            continue
        relative = path.relative_to(package_root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def git_provenance(repo_root: Path) -> dict[str, Any]:
    commit = run_git(repo_root, "rev-parse", "HEAD")
    branch = run_git(repo_root, "branch", "--show-current")
    status = run_git(repo_root, "status", "--porcelain", "--untracked-files=all")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def run_git(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()
