"""Fix labeler stub (SDK removed).

MIGRATION NOTE:
RiskLabeler and SDK labeling tools have been removed.
Use SimpleLabeler patterns from Report_Generator_AI_Threats/src/labeling.py.
"""

from __future__ import annotations


class RiskLabeler:
    """Deprecated — removed with SDK."""
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RiskLabeler removed with SDK. "
            "Use SimpleLabeler pattern from Report_Generator_AI_Threats/src/labeling.py."
        )
