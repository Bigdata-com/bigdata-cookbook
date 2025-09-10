
import os
import pandas as pd
import pickle
from typing import Dict, List, Optional

from src.mindmap.themes import generate_themes

def get_most_granular_elements(tree, element):
    """
    Extracts the elements (labels or summaries) of the most granular (leaf) nodes from the taxonomy tree
    and formats them as a string list.

    Args:
        tree (dict): The taxonomy tree structure with 'Label' and 'Children'.
        element (str): The element of the tree, either 'Label' or 'Summary'

    Returns:
        str: A formatted string with each granular label prefixed by a dash.
    """
    granular_labels = []

    def traverse(node):
        # If the node has no children, it's a leaf node
        if not node.get('Children'):
            sentence = f"{node.get(element, '')}"
            granular_labels.append(sentence)
        else:
            for child in node['Children']:
                traverse(child)

    traverse(tree)

    # Format the labels as a string list
    formatted_labels = [label for label in granular_labels]
    return formatted_labels


def get_label_dict_from_tree(tree):
    """
    Extracts the elements (labels and summaries) of the most granular (leaf) nodes from the taxonomy tree
    and formats them as a string list.

    Args:
        tree (dict): The taxonomy tree structure with 'Label' and 'Children'.

    Returns:
        dict: A dict of Label (keys) and Summary (values)
    """
    granular_dict = {}

    def traverse(node):
        # If the node has no children, it's a leaf node
        if not node.get('Children'):
            label = f"{node.get('Label', '')}"
            summary = f"{node.get('Summary', '')}"
            granular_dict[label] = summary
        else:
            for child in node['Children']:
                traverse(child)

    traverse(tree)

    return granular_dict

    
def generate_themes_tree_dict(general_focus, list_specific_themes, import_from_path: Optional[str] = None, export_to_path: Optional[str] = None):

    # Import Pickle if path provided and file exists
    if import_from_path:
        if os.path.isfile(import_from_path):
            with open(import_from_path, 'rb') as handle:
                dict_themes = pickle.load(handle)
            return dict_themes
    
    dict_themes = {}
    for spec_theme in list_specific_themes:

        MAIN_THEME = spec_theme
        FOCUS = general_focus
        theme_tree = generate_themes(main_theme=MAIN_THEME, focus=FOCUS)
        dict_themes[spec_theme] = theme_tree

    # Export to Pickle if path provided
    if export_to_path:
        with open(export_to_path, 'wb') as handle:
            pickle.dump(dict_themes, handle, protocol=pickle.HIGHEST_PROTOCOL)     
        
    return dict_themes