"""Sequential multi-agent code generation for BFTS nodes."""

from __future__ import annotations

from typing import Any

from rich import print

from .backend import query
from .code_validation import validate_generated_code
from .utils.response import extract_code, extract_text_up_to_code, wrap_code


REVIEWER_FOCI = {
    "PackageReviewer": [
        "missing imports, wrong import paths, and package/API hallucinations",
        "optional dependency use without guards",
        "standard-library/API typos such as wrong keyword names",
    ],
    "DataReviewer": [
        "invented dataset paths and failure to use prepared input/ directories",
        "synthetic-only validation or random data used as main evidence",
        "data/mask shape, dtype, normalization, and train/validation split issues",
    ],
    "TorchShapeReviewer": [
        "model input/output tensor shape compatibility",
        "loss target shape/dtype mismatches",
        "device placement, DataLoader batch handling, and tiny smoke-test feasibility",
    ],
    "MetricReviewer": [
        "missing or misleading evaluation metrics",
        "failure to save working/experiment_data.npy",
        "missing AI_SCIENTIST_SMOKE_TEST branch or incomplete smoke-test output",
        "runtime feasibility within the configured timeout",
    ],
}


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


class SequentialCodeMultiAgent:
    """Planner -> writer -> reviewer -> repairer loop for generated code."""

    def __init__(self, cfg: Any, workspace_dir: Any = None):
        self.cfg = cfg
        self.code_cfg = cfg.agent.code
        self.workspace_dir = workspace_dir or cfg.workspace_dir
        self.multi_cfg = _cfg_get(self.code_cfg, "sequential_multi", {}) or {}
        self.max_review_rounds = int(_cfg_get(self.multi_cfg, "max_review_rounds", 1))
        self.max_repair_rounds = int(_cfg_get(self.multi_cfg, "max_repair_rounds", 3))
        self.run_py_compile = bool(_cfg_get(self.multi_cfg, "run_py_compile", True))
        self.run_import_check = bool(_cfg_get(self.multi_cfg, "run_import_check", True))
        self.reject_synthetic_only = bool(
            _cfg_get(self.multi_cfg, "reject_synthetic_only", True)
        )
        self.run_smoke_test = bool(_cfg_get(self.multi_cfg, "run_smoke_test", False))
        self.require_experiment_data = bool(
            _cfg_get(self.multi_cfg, "require_experiment_data", False)
        )
        self.smoke_test_timeout = int(_cfg_get(self.multi_cfg, "smoke_test_timeout", 60))
        self.reviewers = list(
            _cfg_get(
                self.multi_cfg,
                "reviewers",
                [
                    "PackageReviewer",
                    "DataReviewer",
                    "TorchShapeReviewer",
                    "MetricReviewer",
                ],
            )
        )

    def _query(self, role: str, system_message: Any, user_message: Any = None) -> str:
        print(f"[cyan]SequentialCodeMultiAgent: {role}[/cyan]")
        return query(
            system_message=system_message,
            user_message=user_message,
            model=self.code_cfg.model,
            temperature=self.code_cfg.temp,
            max_tokens=self.code_cfg.max_tokens,
        )

    def _extract_or_repairable_code(self, response: str) -> tuple[str, str]:
        code = extract_code(response)
        plan = extract_text_up_to_code(response)
        if code:
            return plan or "Model returned code without a separate plan.", code
        return (
            "Model failed to return a Python code block.",
            "raise RuntimeError('Sequential code agent failed to return code.')",
        )

    def _planner_prompt(self, base_prompt: Any) -> dict:
        return {
            "Role": "You are CodePlannerAgent.",
            "Task": (
                "Read the research/code-generation request and produce a concise, "
                "implementation-oriented plan. Do not write code. Break the work into "
                "small validated steps and identify likely failure points."
            ),
            "Original request": base_prompt,
            "Output format": "Return concise bullet points only.",
        }

    def _writer_prompt(self, base_prompt: Any, plan: str) -> dict:
        prompt = {
            "Role": "You are CodeWriterAgent.",
            "Planning notes": plan,
            "Original request": base_prompt,
            "Additional requirements": [
                "Return one complete executable Python code block.",
                "Preserve the original request's metric-saving and runtime requirements.",
                "Avoid optional packages unless they are clearly available from the runtime package guidance.",
                "Do not validate the core research claim using only synthetic data.",
                "Include a fast smoke-test branch guarded by os.environ.get('AI_SCIENTIST_SMOKE_TEST') == '1'.",
                "In smoke-test mode, validate dataset paths, load one tiny batch if data exists, run one model forward or one tiny train step, save working/experiment_data.npy, print SMOKE_TEST_PASS, and exit before full training.",
            ],
        }
        return prompt

    def _reviewer_prompt(
        self, reviewer_name: str, base_prompt: Any, plan: str, code: str
    ) -> dict:
        focus = REVIEWER_FOCI.get(
            reviewer_name,
            [
                "real bugs",
                "experiment-invalidating issues",
                "runtime feasibility",
            ],
        )
        return {
            "Role": f"You are {reviewer_name}.",
            "Task": (
                "Review this generated research experiment code before execution. "
                "Be strict, but only report issues that are likely to break execution, "
                "invalidate the experiment, or waste substantial runtime."
            ),
            "Original request": base_prompt,
            "Planning notes": plan,
            "Code to review": wrap_code(code),
            "Primary focus": focus,
            "Output format": (
                "Return REVIEW_PASS if no material issues remain. Otherwise return "
                "a concise numbered list of required fixes. Do not comment on style."
            ),
        }

    def _run_reviewers(self, base_prompt: Any, plan: str, code: str) -> str:
        feedback_parts = []
        for reviewer_name in self.reviewers:
            feedback = self._query(
                reviewer_name,
                self._reviewer_prompt(reviewer_name, base_prompt, plan, code),
            )
            feedback_parts.append(f"## {reviewer_name}\n{feedback.strip()}")
        combined = "\n\n".join(feedback_parts)
        if all("REVIEW_PASS" in part.upper() for part in feedback_parts):
            return "REVIEW_PASS\n\n" + combined
        return combined

    @staticmethod
    def _all_reviewers_passed(review_feedback: str) -> bool:
        return review_feedback.lstrip().upper().startswith("REVIEW_PASS")

    def _repair_prompt(
        self,
        base_prompt: Any,
        plan: str,
        code: str,
        feedback: str,
        validation_feedback: str | None = None,
    ) -> dict:
        return {
            "Role": "You are CodeRepairAgent.",
            "Task": (
                "Revise the code to address the reviewer and validation feedback. "
                "Prefer small targeted fixes. Return the full corrected Python script."
            ),
            "Original request": base_prompt,
            "Planning notes": plan,
            "Previous code": wrap_code(code),
            "Reviewer feedback": feedback,
            "Validation feedback": validation_feedback or "No validation feedback yet.",
            "Output format": "Return a brief repair summary followed by one Python code block.",
        }

    def run(self, base_prompt: Any, retries: int = 3) -> tuple[str, str]:
        plan = self._query("planning", self._planner_prompt(base_prompt))
        writer_response = self._query("writing", self._writer_prompt(base_prompt, plan))
        writer_plan, code = self._extract_or_repairable_code(writer_response)
        combined_plan = f"{plan}\n\nInitial writer notes:\n{writer_plan}".strip()

        review_feedback = ""
        for review_round in range(self.max_review_rounds):
            print(
                "[cyan]SequentialCodeMultiAgent: "
                f"review round {review_round + 1}/{self.max_review_rounds}[/cyan]"
            )
            review_feedback = self._run_reviewers(base_prompt, combined_plan, code)
            if self._all_reviewers_passed(review_feedback):
                break
            repair_response = self._query(
                f"review repair {review_round + 1}/{self.max_review_rounds}",
                self._repair_prompt(base_prompt, combined_plan, code, review_feedback),
            )
            repair_plan, code = self._extract_or_repairable_code(repair_response)
            combined_plan = f"{combined_plan}\n\nReview repair notes:\n{repair_plan}".strip()

        validation_feedback = ""
        for repair_round in range(self.max_repair_rounds + 1):
            validation = validate_generated_code(
                code,
                run_py_compile=self.run_py_compile,
                run_import_check=self.run_import_check,
                reject_synthetic_only=self.reject_synthetic_only,
                run_smoke_test=self.run_smoke_test,
                require_experiment_data=self.require_experiment_data,
                workspace_dir=self.workspace_dir,
                smoke_test_timeout=self.smoke_test_timeout,
            )
            validation_feedback = validation.to_feedback()
            print(f"[cyan]{validation_feedback}[/cyan]")
            if validation.ok:
                if validation.warnings:
                    combined_plan = (
                        f"{combined_plan}\n\nValidation warnings:\n"
                        + "\n".join(validation.warnings)
                    )
                return combined_plan, code

            if repair_round >= self.max_repair_rounds:
                break

            repair_response = self._query(
                f"validation repair {repair_round + 1}/{self.max_repair_rounds}",
                self._repair_prompt(
                    base_prompt,
                    combined_plan,
                    code,
                    review_feedback or "No reviewer feedback.",
                    validation_feedback,
                ),
            )
            repair_plan, code = self._extract_or_repairable_code(repair_response)
            combined_plan = (
                f"{combined_plan}\n\nValidation repair notes:\n{repair_plan}"
            ).strip()

        return (
            combined_plan
            + "\n\nSequential multi-agent validation failed before BFTS execution:\n"
            + validation_feedback,
            "raise RuntimeError('Sequential multi-agent code validation failed.')",
        )
