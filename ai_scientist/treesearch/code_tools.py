"""Restricted read-only tool loop for generated code agents."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from rich import print

from .backend import query


DEFAULT_IMPORT_CHECKS = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "torch",
    "torchvision",
    "PIL",
    "yaml",
    "psutil",
    "openai",
    "timm",
    "albumentations",
    "clip",
]


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trim(text: Any, max_chars: int = 6000) -> str:
    compact = str(text)
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "\n...<truncated>..."


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class ToolUsingCodeAgent:
    """Collect local context through a restricted JSON-action tool loop."""

    def __init__(self, cfg: Any, workspace_dir: Any = None):
        self.cfg = cfg
        self.code_cfg = cfg.agent.code
        self.multi_cfg = _cfg_get(self.code_cfg, "sequential_multi", {}) or {}
        self.tool_cfg = _cfg_get(self.multi_cfg, "tool_loop", {}) or {}
        self.workspace_dir = Path(workspace_dir or cfg.workspace_dir).resolve()
        self.repo_root = Path(
            os.environ.get("AI_SCIENTIST_ROOT", Path.cwd())
        ).resolve()
        self.max_tool_steps = int(_cfg_get(self.tool_cfg, "max_tool_steps", 8))
        self.max_read_chars = int(_cfg_get(self.tool_cfg, "max_read_chars", 8000))
        self.max_rg_results = int(_cfg_get(self.tool_cfg, "max_rg_results", 30))
        self.allowed_tools = set(
            _cfg_get(
                self.tool_cfg,
                "allowed_tools",
                [
                    "list_files",
                    "read_file",
                    "rg",
                    "inspect_input",
                    "inspect_requirements",
                    "inspect_imports",
                    "finish",
                ],
            )
        )
        self.transcript: list[str] = []

    def _query(self, system_message: Any, user_message: Any = None) -> str:
        return query(
            system_message=system_message,
            user_message=user_message,
            model=self.code_cfg.model,
            temperature=self.code_cfg.temp,
            max_tokens=self.code_cfg.max_tokens,
        )

    def _resolve_safe_path(self, path: str | None) -> Path:
        raw = Path(path or ".")
        if not raw.is_absolute():
            candidate = (self.repo_root / raw).resolve()
            if not self._is_allowed_path(candidate):
                candidate = (self.workspace_dir / raw).resolve()
        else:
            candidate = raw.resolve()
        if not self._is_allowed_path(candidate):
            raise ValueError(f"Path is outside allowed roots: {path}")
        return candidate

    def _is_allowed_path(self, path: Path) -> bool:
        roots = [self.repo_root, self.workspace_dir]
        for root in roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _tool_list_files(self, args: dict) -> str:
        path = self._resolve_safe_path(args.get("path", "."))
        max_entries = int(args.get("max_entries", 80))
        if not path.exists():
            return f"{path} does not exist."
        if path.is_file():
            return f"{path} is a file ({path.stat().st_size} bytes)."
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))[
            :max_entries
        ]:
            kind = "dir" if child.is_dir() else "file"
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{kind}: {child.name}{suffix}")
        return "\n".join(entries) or f"{path} is empty."

    def _tool_read_file(self, args: dict) -> str:
        path = self._resolve_safe_path(args.get("path"))
        start = max(int(args.get("start", 1)), 1)
        end = args.get("end")
        end = int(end) if end is not None else None
        if not path.exists():
            return f"{path} does not exist."
        if not path.is_file():
            return f"{path} is not a file."
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[start - 1 : end]
        numbered = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start)]
        return _trim("\n".join(numbered), self.max_read_chars)

    def _tool_rg(self, args: dict) -> str:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return "rg requires a non-empty pattern."
        path = self._resolve_safe_path(args.get("path", "."))
        if not path.exists():
            return f"{path} does not exist."
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--max-count",
            "5",
            pattern,
            str(path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
            )
        except FileNotFoundError:
            return "rg is not installed."
        except subprocess.TimeoutExpired:
            return "rg timed out."
        lines = (proc.stdout or "").splitlines()[: self.max_rg_results]
        return "\n".join(lines) if lines else "No matches."

    def _tool_inspect_input(self, args: dict) -> str:
        input_dir = self.workspace_dir / "input"
        if not input_dir.exists():
            return f"No input directory found at {input_dir}."
        lines = [f"Observed input root: {input_dir}"]
        for root, dirs, files in os.walk(input_dir, followlinks=True):
            root_path = Path(root)
            depth = len(root_path.relative_to(input_dir).parts)
            if depth > 2:
                dirs[:] = []
                continue
            rel = root_path.relative_to(input_dir)
            rel_name = "." if str(rel) == "." else str(rel)
            ext_counts: dict[str, int] = {}
            for file_name in files:
                ext = Path(file_name).suffix.lower() or "<no_ext>"
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            count_text = ", ".join(
                f"{ext}:{count}" for ext, count in sorted(ext_counts.items())
            )
            lines.append(
                f"{rel_name}: {len(files)} files"
                + (f" ({count_text})" if count_text else "")
            )
        return _trim("\n".join(lines), 8000)

    def _tool_inspect_requirements(self, args: dict) -> str:
        candidates = [
            "requirements.txt",
            "pyproject.toml",
            "environment.yml",
            "environment.yaml",
            "setup.py",
        ]
        parts = []
        for rel_path in candidates:
            path = self.repo_root / rel_path
            if path.exists() and path.is_file():
                parts.append(f"## {rel_path}\n" + _trim(path.read_text(errors="replace"), 3000))
        return "\n\n".join(parts) if parts else "No common dependency files found."

    def _tool_inspect_imports(self, args: dict) -> str:
        modules = args.get("modules") or DEFAULT_IMPORT_CHECKS
        if isinstance(modules, str):
            modules = [m.strip() for m in modules.split(",") if m.strip()]
        lines = []
        for module_name in modules:
            available = importlib.util.find_spec(str(module_name)) is not None
            lines.append(f"{module_name}: {'available' if available else 'missing'}")
        return "\n".join(lines)

    def _run_tool(self, action: str, args: dict) -> str:
        if action not in self.allowed_tools:
            return f"Tool {action} is not allowed."
        if action == "list_files":
            return self._tool_list_files(args)
        if action == "read_file":
            return self._tool_read_file(args)
        if action == "rg":
            return self._tool_rg(args)
        if action == "inspect_input":
            return self._tool_inspect_input(args)
        if action == "inspect_requirements":
            return self._tool_inspect_requirements(args)
        if action == "inspect_imports":
            return self._tool_inspect_imports(args)
        if action == "finish":
            return "Finished tool context collection."
        return f"Unknown tool: {action}"

    def _system_prompt(self, base_prompt: Any) -> dict:
        return {
            "Role": "You are RepoToolAgent.",
            "Task": (
                "Use the allowed read-only tools to gather only the local context "
                "needed before code generation. Prefer observed facts over guesses."
            ),
            "Original code-generation request": base_prompt,
            "Allowed tools": sorted(self.allowed_tools),
            "Action format": (
                "Return exactly one JSON object per turn: "
                '{"action": "tool_name", "args": {...}, "reason": "short reason"}. '
                "Use finish when enough context has been gathered."
            ),
            "Important limits": [
                "Do not request arbitrary shell commands.",
                "Do not assume files exist unless tool output showed them.",
                "Use paths relative to the repo root or workspace.",
            ],
        }

    def _user_prompt(self) -> str:
        if not self.transcript:
            return (
                "Start by inspecting input data, dependency files, and relevant repo "
                "files needed for generated experiment code."
            )
        return "Tool transcript so far:\n" + "\n\n".join(self.transcript[-8:])

    def run(self, base_prompt: Any) -> str:
        for step in range(self.max_tool_steps):
            print(
                "[cyan]ToolUsingCodeAgent: "
                f"tool step {step + 1}/{self.max_tool_steps}[/cyan]"
            )
            response = self._query(self._system_prompt(base_prompt), self._user_prompt())
            action_obj = _extract_json_object(response)
            if not action_obj:
                self.transcript.append(
                    "Invalid tool action response; expected a JSON object."
                )
                continue

            action = str(action_obj.get("action", "")).strip()
            args = action_obj.get("args") or {}
            reason = str(action_obj.get("reason", "")).strip()
            if not isinstance(args, dict):
                args = {}
            try:
                result = self._run_tool(action, args)
            except Exception as exc:
                result = f"Tool error: {exc}"
            record = (
                f"Step {step + 1}: action={action}, reason={reason}\n"
                f"args={json.dumps(args, ensure_ascii=False)}\n"
                f"result:\n{_trim(result, 6000)}"
            )
            self.transcript.append(record)
            if action == "finish":
                break

        if not self.transcript:
            return "No tool context collected."
        header = (
            "Observed local tool context for this run. This is a snapshot, not a "
            "permanent guarantee. If a file or directory is not listed here, do not "
            "assume it exists."
        )
        return header + "\n\n" + "\n\n".join(self.transcript)
