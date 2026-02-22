import ast
from pathlib import Path


def _collect_parser_flags(module: ast.Module) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(module):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                flags.add(arg.value)
    return flags


def _collect_override_flags(module: ast.Module) -> set[str]:
    flags: set[str] = set()
    for node in module.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "_apply_cli_overrides"):
            continue

        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "overrides":
                overrides = stmt.value
                if isinstance(overrides, ast.List):
                    for row in overrides.elts:
                        if not (isinstance(row, ast.Tuple) and row.elts):
                            continue
                        flag_list = row.elts[0]
                        if isinstance(flag_list, ast.List):
                            for f in flag_list.elts:
                                if isinstance(f, ast.Constant) and isinstance(f.value, str):
                                    flags.add(f.value)

            # Also include explicit boolean convenience flags handled via _flag_in_argv.
            if isinstance(stmt, ast.If):
                for inner in ast.walk(stmt.test):
                    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)):
                        continue
                    if inner.func.id != "_flag_in_argv" or not inner.args:
                        continue
                    flag_arg = inner.args[0]
                    if isinstance(flag_arg, ast.List):
                        for f in flag_arg.elts:
                            if isinstance(f, ast.Constant) and isinstance(f.value, str):
                                flags.add(f.value)

    return flags


def test_all_runtime_cli_flags_have_toml_mapping():
    train_dynamic = Path(__file__).resolve().parents[1] / "df_mlx" / "train_dynamic.py"
    src = train_dynamic.read_text(encoding="utf-8")
    module = ast.parse(src)

    parser_flags = _collect_parser_flags(module)
    mapped_flags = _collect_override_flags(module)

    # These control how the TOML is loaded/printed and are intentionally CLI-only.
    meta_only_flags = {"--run-config", "--print-run-config", "--preset"}

    missing = sorted(parser_flags - mapped_flags - meta_only_flags)
    assert missing == [], f"CLI flags missing TOML mapping: {missing}"

    stale = sorted(mapped_flags - parser_flags)
    assert stale == [], f"TOML mapping contains unknown CLI flags: {stale}"
