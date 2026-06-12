"""Analysis-mode profiles for the screener pipeline.

A *mode* selects the prompts, defaults, and leaf-label semantics used by the
pipeline without changing any of its mechanics. Two modes are supported:

``thematic-screener``
    The original behavior: decompose a *theme* into sub-themes and screen
    companies for thematic exposure.
``risk-analyzer``
    Decompose a *risk* into a risk-channel/risk-factor/sub-scenario taxonomy and
    screen companies for risk exposure, producing output compatible with the
    Bigdata Risk Analyzer app schema.

Each :class:`ModeProfile` bundles everything mode-specific so the rest of the
pipeline can stay agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.prompts import (
    RISK_SUMMARY_TEMPLATE,
    SYSTEM_MESSAGE_LABELS,
    SYSTEM_MESSAGE_RISK,
    SYSTEM_PROMPT_LABELING,
    SYSTEM_PROMPT_RISK_LABELING,
    THEMATIC_SUMMARY_TEMPLATE,
    USER_MESSAGE_LABELS,
    USER_MESSAGE_RISK,
)


class AnalysisMode(str, Enum):
    """Supported analysis modes."""

    THEMATIC_SCREENER = "thematic-screener"
    RISK_ANALYZER = "risk-analyzer"


class LeafField(str, Enum):
    """Which taxonomy field is used as the working label for leaf nodes."""

    SUMMARY = "summary"
    LABEL = "label"


@dataclass(frozen=True)
class ModeProfile:
    """Bundle of mode-specific prompts, defaults, and label semantics.

    Attributes
    ----------
    mode:
        The :class:`AnalysisMode` this profile implements.
    labels_system_prompt:
        System prompt for taxonomy generation. Uses ``{main_theme}`` and
        ``{analyst_focus}`` placeholders.
    labels_user_prompt:
        User prompt for taxonomy generation. Uses the ``{main_theme}``
        placeholder.
    labeling_system_prompt:
        System prompt for sentence labeling. Uses ``{main_theme}`` and
        ``{labels}`` placeholders.
    company_summary_template:
        System prompt template for company summaries. Uses the ``{main_theme}``
        placeholder.
    default_main_theme:
        Default main concept (theme or risk) when none is supplied.
    default_analyst_focus:
        Default analyst focus when none is supplied.
    leaf_field:
        Which taxonomy field (``summary`` or ``label``) is used as the working
        label set returned to the rest of the pipeline.
    """

    mode: AnalysisMode
    labels_system_prompt: str
    labels_user_prompt: str
    labeling_system_prompt: str
    company_summary_template: str
    default_main_theme: str
    default_analyst_focus: str
    leaf_field: LeafField


_THEMATIC_PROFILE = ModeProfile(
    mode=AnalysisMode.THEMATIC_SCREENER,
    labels_system_prompt=SYSTEM_MESSAGE_LABELS,
    labels_user_prompt=USER_MESSAGE_LABELS,
    labeling_system_prompt=SYSTEM_PROMPT_LABELING,
    company_summary_template=THEMATIC_SUMMARY_TEMPLATE,
    default_main_theme="AI disruption in product development",
    default_analyst_focus="How companies are including AI in their development cycle",
    leaf_field=LeafField.LABEL,
)


_RISK_PROFILE = ModeProfile(
    mode=AnalysisMode.RISK_ANALYZER,
    labels_system_prompt=SYSTEM_MESSAGE_RISK,
    labels_user_prompt=USER_MESSAGE_RISK,
    labeling_system_prompt=SYSTEM_PROMPT_RISK_LABELING,
    company_summary_template=RISK_SUMMARY_TEMPLATE,
    default_main_theme="US Government Shutdown",
    default_analyst_focus="How a prolonged federal funding lapse affects company operations and revenue",
    leaf_field=LeafField.LABEL,
)


MODE_PROFILES: dict[AnalysisMode, ModeProfile] = {
    AnalysisMode.THEMATIC_SCREENER: _THEMATIC_PROFILE,
    AnalysisMode.RISK_ANALYZER: _RISK_PROFILE,
}


def get_profile(mode: AnalysisMode | str) -> ModeProfile:
    """Return the :class:`ModeProfile` for ``mode``.

    Accepts either an :class:`AnalysisMode` or its string value so callers can
    pass values read straight from the persisted config.
    """
    resolved = AnalysisMode(mode) if not isinstance(mode, AnalysisMode) else mode
    return MODE_PROFILES[resolved]
