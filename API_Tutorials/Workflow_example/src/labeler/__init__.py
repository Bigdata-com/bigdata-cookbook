"""Labeler module (SDK removed — use OpenAI-based labelers instead).

MIGRATION NOTE:
NarrativeLabeler and other SDK labelers have been removed.
Use SimpleLabeler patterns from Report_Generator_AI_Threats/src/labeling.py
or implement custom OpenAI-based labeling.
"""

from __future__ import annotations


def NarrativeLabeler(*args, **kwargs):
    """Deprecated — removed with SDK."""
    raise NotImplementedError(
        "NarrativeLabeler removed with SDK. "
        "Use SimpleLabeler pattern from Report_Generator_AI_Threats/src/labeling.py "
        "or implement custom OpenAI-based labeling."
    )
