import ast
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_PKG_ROOT = Path(__file__).resolve().parents[1]
DF_DIR = REPO_PKG_ROOT / "df"
DF_MLX_DIR = REPO_PKG_ROOT / "df_mlx"


def _iter_py_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _collect_import_targets(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                targets.append(node.module)
    return targets


def _find_cross_imports(base: Path, forbidden_root: str) -> List[Tuple[Path, str]]:
    violations: List[Tuple[Path, str]] = []
    prefix = f"{forbidden_root}."
    for path in _iter_py_files(base):
        for target in _collect_import_targets(path):
            if target == forbidden_root or target.startswith(prefix):
                violations.append((path, target))
    return violations


def test_df_mlx_does_not_import_df() -> None:
    violations = _find_cross_imports(DF_MLX_DIR, "df")
    assert not violations, f"df_mlx must not import df modules: {violations}"


def test_df_does_not_import_df_mlx() -> None:
    violations = _find_cross_imports(DF_DIR, "df_mlx")
    assert not violations, f"df must not import df_mlx modules: {violations}"
