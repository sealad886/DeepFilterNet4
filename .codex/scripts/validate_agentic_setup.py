#!/usr/bin/env python3
"""Validate repo-local Codex setup references and config wiring."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / ".codex"
CONFIG_PATH = CODEX_DIR / "config.toml"
MARKDOWN_PATHS = sorted(CODEX_DIR.rglob("*.md"))
FILE_REF_RE = re.compile(r"#file:([^\s`]+)")
SKILL_REF_RE = re.compile(r"#skill:([A-Za-z0-9_-]+)")
NAME_RE = re.compile(r'^name:\s*(?:"([^"]+)"|\'([^\']+)\'|([^\n#]+))$', re.MULTILINE)


def parse_frontmatter_name(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    frontmatter = text[4:end]
    match = NAME_RE.search(frontmatter)
    if not match:
        return None
    value = next(group for group in match.groups() if group is not None)
    return value.strip()


def parse_simple_value(raw_value: str):
    if raw_value.startswith('"') and raw_value.endswith('"'):
        return raw_value[1:-1]
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    if raw_value.startswith("[") and raw_value.endswith("]"):
        return re.findall(r'"([^"]*)"', raw_value)
    if raw_value.lstrip("-").isdigit():
        return int(raw_value)
    return raw_value


def load_config() -> dict:
    text = CONFIG_PATH.read_text(encoding="utf-8")

    try:
        import tomllib  # type: ignore[attr-defined]

        return tomllib.loads(text)
    except ModuleNotFoundError:
        pass

    data: dict = {}
    current = data
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            current = data.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        current[key.strip()] = parse_simple_value(raw_value.strip())
    return data


def resolve_file_ref(source: Path, raw_ref: str) -> Path:
    ref = Path(raw_ref)
    if ref.is_absolute():
        return ref
    return (source.parent / ref).resolve()


def validate_config(errors: list[str]) -> None:
    try:
        config = load_config()
    except OSError as exc:
        errors.append(f"Failed to parse {CONFIG_PATH.relative_to(ROOT)}: {exc}")
        return

    model_instructions = config.get("model_instructions_file")
    if not model_instructions:
        errors.append("config.toml is missing model_instructions_file")
    else:
        model_path = (CONFIG_PATH.parent / model_instructions).resolve()
        if not model_path.exists():
            errors.append("Missing model instructions file relative to .codex/config.toml: " f"{model_instructions}")

    fallback_paths = config.get("project_doc_fallback_filenames", [])
    if not isinstance(fallback_paths, list):
        errors.append("project_doc_fallback_filenames must be a list")
    else:
        for raw_path in fallback_paths:
            if not isinstance(raw_path, str):
                errors.append("project_doc_fallback_filenames must contain strings")
                continue
            candidate = (ROOT / raw_path).resolve()
            if not candidate.exists():
                errors.append(f"Missing fallback project doc: {raw_path}")

    features = config.get("features", {})
    if features.get("multi_agent") is not True:
        errors.append("features.multi_agent must be true for this repo setup")

    agents = config.get("agents", {})
    for key in ("max_threads", "max_depth", "job_max_runtime_seconds"):
        value = agents.get(key)
        if not isinstance(value, int) or value < 1:
            errors.append(f"agents.{key} must be a positive integer")


def collect_skill_names(errors: list[str]) -> set[str]:
    skill_names: set[str] = set()
    for skill_file in sorted((CODEX_DIR / "skills").glob("*/SKILL.md")):
        name = parse_frontmatter_name(skill_file)
        if not name:
            errors.append(f"Missing skill name frontmatter: {skill_file.relative_to(ROOT)}")
            continue
        if name in skill_names:
            errors.append(f"Duplicate skill name: {name}")
            continue
        skill_names.add(name)
    return skill_names


def validate_markdown_refs(skill_names: set[str], errors: list[str]) -> None:
    for path in MARKDOWN_PATHS:
        text = path.read_text(encoding="utf-8")
        for raw_ref in FILE_REF_RE.findall(text):
            resolved = resolve_file_ref(path, raw_ref)
            if not resolved.exists():
                errors.append(f"Missing #file target in {path.relative_to(ROOT)}: {raw_ref}")
        for raw_skill in SKILL_REF_RE.findall(text):
            if raw_skill not in skill_names:
                errors.append(f"Missing #skill target in {path.relative_to(ROOT)}: {raw_skill}")


def validate_frontmatter_names(errors: list[str]) -> None:
    seen: dict[str, Path] = {}
    for path in MARKDOWN_PATHS:
        if path.name == "README.md" or path.name == "AGENTS.md" or path.name == "TOOLSET.md":
            continue
        name = parse_frontmatter_name(path)
        if not name:
            errors.append(f"Missing name frontmatter: {path.relative_to(ROOT)}")
            continue
        other = seen.get(name)
        if other is not None:
            errors.append(
                "Duplicate frontmatter name " f"{name}: {other.relative_to(ROOT)} and {path.relative_to(ROOT)}"
            )
            continue
        seen[name] = path


def main() -> int:
    errors: list[str] = []
    validate_config(errors)
    skill_names = collect_skill_names(errors)
    validate_frontmatter_names(errors)
    validate_markdown_refs(skill_names, errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("DeepFilterNet Codex setup validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
