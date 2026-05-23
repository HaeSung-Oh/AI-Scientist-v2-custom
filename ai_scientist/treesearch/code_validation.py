"""Validation helpers for generated experiment code."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass


@dataclass
class CodeValidationResult:
    ok: bool
    checks: list[str]
    errors: list[str]
    warnings: list[str]

    def to_feedback(self) -> str:
        lines = ["Validation result: " + ("PASS" if self.ok else "FAIL")]
        if self.checks:
            lines.append("Checks:")
            lines.extend(f"- {item}" for item in self.checks)
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"- {item}" for item in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines)


def _top_level_imports(tree: ast.AST) -> set[str]:
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".")[0])
    return imports


def _is_stdlib_or_local(module_name: str) -> bool:
    if module_name in sys.stdlib_module_names:
        return True
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False
    origin = spec.origin or ""
    return "site-packages" not in origin and "dist-packages" not in origin


def validate_generated_code(
    code: str,
    *,
    run_py_compile: bool = True,
    run_import_check: bool = True,
    reject_synthetic_only: bool = True,
) -> CodeValidationResult:
    checks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    tree = None

    if run_py_compile:
        try:
            tree = ast.parse(code)
            compile(code, "generated_experiment.py", "exec")
            checks.append("Python syntax compile passed.")
        except SyntaxError as exc:
            errors.append(f"SyntaxError line {exc.lineno}: {exc.msg}")
            return CodeValidationResult(False, checks, errors, warnings)

    if tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

    if run_import_check and tree is not None:
        missing = []
        for module_name in sorted(_top_level_imports(tree)):
            if _is_stdlib_or_local(module_name):
                continue
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
        if missing:
            errors.append(
                "Missing importable top-level modules: " + ", ".join(missing)
            )
        else:
            checks.append("Top-level import availability check passed.")

    if reject_synthetic_only:
        synthetic_markers = [
            r"class\s+Synthetic",
            r"SyntheticDataset",
            r"torch\.rand\s*\(",
            r"torch\.randn\s*\(",
            r"np\.random\.(rand|randn|random)",
            r"procedurally generated",
            r"simulated data",
        ]
        hits = [
            marker
            for marker in synthetic_markers
            if re.search(marker, code, flags=re.IGNORECASE)
        ]
        if hits:
            warnings.append(
                "Synthetic-data markers found. This is allowed only for smoke tests, "
                "not final validation: "
                + ", ".join(hits)
            )
        else:
            checks.append("No obvious synthetic-only data markers found.")

    return CodeValidationResult(not errors, checks, errors, warnings)
