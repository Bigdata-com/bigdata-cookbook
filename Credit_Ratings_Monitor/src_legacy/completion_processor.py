import json
import re

import pandas as pd

def completion_to_dataframe(completion: dict) -> pd.DataFrame:
    """
    Converts a completion dictionary into a structured pandas DataFrame.
    
    Args:
    - completion (dict): A dictionary containing completion data from a model.
    
    Returns:
    - pd.DataFrame: A pandas DataFrame containing structured completion data.
    """
        
    completion_message_content = completion.get('choices', [{}])[0].get('message', {}).get('content', {})
    completion_finish_reason = completion.get('choices', [{}])[0].get('finish_reason', None)
    completion_usage = completion.get('usage', {})
    completion_id = completion.get('id', None)
    completion_system_fingerprint = completion.get('system_fingerprint', None)
    completion_model = completion.get('model', None)

    if completion_message_content:
        try:
            completion_message_content = re.sub('```', '', completion_message_content)
            completion_message_content = re.sub('json', '', completion_message_content)
            completion_message_content = json.loads(completion_message_content)
        except json.JSONDecodeError:
            completion_message_content = re.sub('```', '', completion_message_content)
            completion_message_content = re.sub('json', '', completion_message_content)
            completion_message_content = json.loads((', \n'.join(re.findall(r'"\d+":\s*{.*?}\s*(?=,\s*"\d+"|$)', completion_message_content, re.DOTALL))).join(['{', '}']).strip().replace("\\'", "'"))
            # ''.join(re.findall(r'"\d+":\s*{.*?}\s*(?=,\s*"\d+"|$)', completion_message_content, re.DOTALL))
            # re.findall(r'"(\d+)"(.*?)\}', completion_message_content, re.DOTALL)
        completion_message_content_dict = {}
        for key, value in completion_message_content.items():
            try:
                if isinstance(value, dict):
                    completion_message_content_dict[int(key)] = value
                else:
                    try:
                        value = int(value)
                        key = int(key)
                    except ValueError:
                        continue
                    completion_message_content_dict[key] = {'label': value}  # value}
            except ValueError:
                pass

    completion_df = pd.DataFrame.from_dict(completion_message_content_dict, orient='index')

    completion_df['model'] = completion_model
    completion_df['finish_reason'] = completion_finish_reason
    completion_df['completion_tokens'] = completion_usage.get('completion_tokens', None)
    completion_df['prompt_tokens'] = completion_usage.get('prompt_tokens', None)
    completion_df['total_tokens'] = completion_usage.get('total_tokens', None)
    completion_df['completion_id'] = completion_id
    completion_df['system_fingerprint'] = completion_system_fingerprint
    return completion_df