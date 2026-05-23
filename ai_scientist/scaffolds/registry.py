"""Select task-adaptive scaffold guidance from observed context."""

from __future__ import annotations

from dataclasses import dataclass

from .guidance import (
    GENERIC_EXPERIMENT_GUIDANCE,
    IMAGE_CLASSIFICATION_GUIDANCE,
    POLYP_SEGMENTATION_GUIDANCE,
    SEGMENTATION_GUIDANCE,
    TABULAR_GUIDANCE,
    TIMESERIES_GUIDANCE,
)


@dataclass(frozen=True)
class Scaffold:
    name: str
    confidence: str
    reason: str
    guidance: str


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def select_scaffold(task_desc: object, tool_context: str) -> Scaffold:
    """Return the best scaffold candidate without forcing domain-specific code."""
    text = f"{task_desc}\n{tool_context}".lower()

    polyp_keywords = [
        "polyp",
        "kvasir",
        "cvc-clinicdb",
        "clinicdb",
        "colon",
        "endoscopy",
        "colonoscopy",
    ]
    segmentation_keywords = [
        "segmentation",
        "mask",
        "masks",
        "dice",
        "iou",
        "pixel",
        "unet",
        "image/mask",
    ]
    timeseries_keywords = [
        "time series",
        "timeseries",
        "temporal",
        "sequence",
        "forecast",
        "forecasting",
        "sensor",
        "ecg",
        "eeg",
        "window",
        "sliding",
        "lag",
    ]
    tabular_keywords = [
        ".csv",
        ".parquet",
        "tabular",
        "table",
        "dataframe",
        "classification",
        "regression",
    ]
    image_classification_keywords = [
        "image classification",
        "classify image",
        "classification dataset",
        "jpg:",
        "jpeg:",
        "png:",
    ]

    if _contains_any(text, polyp_keywords) and _contains_any(
        text, segmentation_keywords
    ):
        return Scaffold(
            name="polyp_segmentation",
            confidence="medium",
            reason="Observed polyp/colonoscopy and segmentation signals.",
            guidance=POLYP_SEGMENTATION_GUIDANCE,
        )

    if _contains_any(text, timeseries_keywords):
        return Scaffold(
            name="generic_timeseries",
            confidence="medium",
            reason="Observed temporal/sequence/forecasting signals.",
            guidance=TIMESERIES_GUIDANCE,
        )

    if _contains_any(text, segmentation_keywords):
        return Scaffold(
            name="generic_segmentation",
            confidence="medium",
            reason="Observed image/mask or segmentation metric signals.",
            guidance=SEGMENTATION_GUIDANCE,
        )

    if _contains_any(text, image_classification_keywords):
        return Scaffold(
            name="generic_image_classification",
            confidence="low",
            reason="Observed image classification signals.",
            guidance=IMAGE_CLASSIFICATION_GUIDANCE,
        )

    if _contains_any(text, tabular_keywords):
        return Scaffold(
            name="generic_tabular",
            confidence="low",
            reason="Observed table-like file or task signals.",
            guidance=TABULAR_GUIDANCE,
        )

    return Scaffold(
        name="generic_experiment",
        confidence="low",
        reason="No specific scaffold matched; using generic experiment guidance.",
        guidance=GENERIC_EXPERIMENT_GUIDANCE,
    )


def format_scaffold_for_prompt(scaffold: Scaffold) -> str:
    return (
        "Task-adaptive scaffold candidate\n"
        f"- name: {scaffold.name}\n"
        f"- confidence: {scaffold.confidence}\n"
        f"- reason: {scaffold.reason}\n\n"
        "Use this scaffold only if it matches the task and observed local context. "
        "If it does not match, explicitly fall back to generic experiment code. "
        "Do not force a domain-specific scaffold onto unrelated tasks.\n\n"
        + scaffold.guidance
    )
