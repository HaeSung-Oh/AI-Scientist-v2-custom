"""Validation helpers for generated experiment code."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


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
    run_smoke_test: bool = False,
    require_experiment_data: bool = False,
    workspace_dir: str | Path | None = None,
    smoke_test_timeout: int = 60,
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

    if run_smoke_test:
        if "AI_SCIENTIST_SMOKE_TEST" not in code:
            errors.append(
                "Smoke test is enabled, but generated code does not check "
                "AI_SCIENTIST_SMOKE_TEST. Add a fast smoke-test branch that "
                "validates data loading, one tiny model forward/train step, and "
                "experiment_data.npy saving before full training."
            )
            return CodeValidationResult(False, checks, errors, warnings)

        smoke_result = run_generated_smoke_test(
            code,
            workspace_dir=workspace_dir,
            timeout=smoke_test_timeout,
            require_experiment_data=require_experiment_data,
        )
        checks.extend(smoke_result.checks)
        errors.extend(smoke_result.errors)
        warnings.extend(smoke_result.warnings)

    return CodeValidationResult(not errors, checks, errors, warnings)


def run_generated_smoke_test(
    code: str,
    *,
    workspace_dir: str | Path | None,
    timeout: int,
    require_experiment_data: bool,
) -> CodeValidationResult:
    checks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    base_workspace = Path(workspace_dir).resolve() if workspace_dir else None
    with tempfile.TemporaryDirectory(prefix="ai_scientist_smoke_") as tmp:
        smoke_dir = Path(tmp)
        runfile = smoke_dir / "runfile.py"
        runfile.write_text(code, encoding="utf-8")
        (smoke_dir / "working").mkdir(exist_ok=True)

        if base_workspace is not None:
            input_src = base_workspace / "input"
            if input_src.exists():
                try:
                    (smoke_dir / "input").symlink_to(input_src, target_is_directory=True)
                except OSError:
                    warnings.append(
                        f"Could not symlink smoke-test input directory from {input_src}."
                    )

        env = os.environ.copy()
        env["AI_SCIENTIST_SMOKE_TEST"] = "1"
        env.setdefault("MPLCONFIGDIR", str(smoke_dir / ".matplotlib"))

        try:
            proc = subprocess.run(
                [sys.executable, str(runfile)],
                cwd=smoke_dir,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or ""
            errors.append(
                f"Smoke test timed out after {timeout}s. Output preview: "
                + output[-2000:]
            )
            return CodeValidationResult(False, checks, errors, warnings)

        output = proc.stdout or ""
        if proc.returncode != 0:
            errors.append(
                f"Smoke test exited with code {proc.returncode}. Output preview: "
                + output[-3000:]
            )
            return CodeValidationResult(False, checks, errors, warnings)

        checks.append("AI_SCIENTIST_SMOKE_TEST execution passed.")

        expected_data = smoke_dir / "working" / "experiment_data.npy"
        if require_experiment_data:
            if expected_data.exists():
                checks.append("Smoke test created working/experiment_data.npy.")
            else:
                errors.append(
                    "Smoke test passed but did not create working/experiment_data.npy."
                )
        elif not expected_data.exists():
            warnings.append("Smoke test did not create working/experiment_data.npy.")

    return CodeValidationResult(not errors, checks, errors, warnings)
