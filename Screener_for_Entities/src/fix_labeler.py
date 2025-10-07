"""
Monkey patch for RiskLabeler to handle malformed responses gracefully.
"""

from bigdata_research_tools.labeler.risk_labeler import RiskLabeler
from pandas import DataFrame
from typing import List, Dict, Any
import warnings


def robust_deserialize_label_responses(self, responses: List[Dict[str, Any]]) -> DataFrame:
    """
    Enhanced version that validates keys before processing and skips invalid ones.
    """
    response_mapping = {}
    skipped_keys = []
    problematic_responses = []
    
    for i, response in enumerate(responses):
        if not response or not isinstance(response, dict):
            continue

        for k, v in response.items():
            # Check if key can be converted to int (sentence ID)
            try:
                sentence_id = int(k)
            except (ValueError, TypeError):
                skipped_keys.append(k)
                # Capture the problematic response for debugging
                problematic_responses.append(f"Response {i}: {response}")
                continue
                
            try:
                response_mapping[k] = {
                    "motivation": v.get("motivation", ""),
                    "label": v.get("label", self.unknown_label),
                    **{
                        key: value
                        for key, value in v.items()
                        if key not in ["motivation", "label"]
                    },
                }
            except (KeyError, AttributeError):
                response_mapping[k] = {
                    "motivation": "",
                    "label": self.unknown_label,
                }

    # Warn about skipped keys and show problematic responses
    if skipped_keys:
        warnings.warn(f"Skipped invalid response keys (not sentence IDs): {skipped_keys}")
        print("=== PROBLEMATIC RESPONSES FOR DEBUGGING ===")
        for prob_resp in problematic_responses[:3]:  # Show first 3 problematic responses
            print(prob_resp)
        print("===========================================")
    
    if not response_mapping:
        # Return empty DataFrame with expected columns if no valid responses
        return DataFrame(columns=["motivation", "label", "sentiment", "quotes"])
    
    df_labels = DataFrame.from_dict(response_mapping, orient="index")
    df_labels.index = df_labels.index.astype(int)  # Safe now since we validated
    return df_labels


# Apply the monkey patch
RiskLabeler._deserialize_label_responses = robust_deserialize_label_responses