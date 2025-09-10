"""
Module that includes all functions to create or extract
information related to the sub-theme tree structure. 

Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Jelena Starovic (jstarovic@ravenpack.com)
"""

import ast
import os
import re
from typing import Dict, Any, List

import openai
import pandas as pd
import plotly.express as px
from src.mindmap.theme_prompts import compose_themes_system_prompt_base

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = 'gpt-4o-mini'

TEMPERATURE = 0.01  # Deterministic as possible
RANDOM_SEED = 42


def generate_themes(main_theme: str,
                    focus: str = '') -> Dict[str, Any]:
    """
    Generate themes based on the main theme.

    :param main_theme: The main theme
    :param focus: The focus(es), if any
    :return: The generated themes
    """
    openai_client = openai.OpenAI()

    system_prompt = compose_themes_system_prompt_base()
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': main_theme
            },
            {
                'role': 'user',
                'content': focus
            }
        ],
        temperature=TEMPERATURE,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=RANDOM_SEED,
        response_format={'type': 'json_object'}
    )

    tree_str = response.model_dump()['choices'][0]['message']['content']
    tree_str = re.sub('```', '', tree_str)
    tree_str = re.sub('json', '', tree_str)

    # Convert string into dictionary
    tree_dict = ast.literal_eval(tree_str)
    return tree_dict


def convert_to_node_tree(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert the tree into a node tree.

    :param tree: The tree
    :return: The node tree
    """
    def convert_(node):
        new_node = {'label': node['Label'],
                    'value': f'node_{node["Node"]}',
                    'summary': node['Summary']}
        if 'Children' in node:
            new_node['children'] = [convert_(child)
                                    for child in node['Children']]
        return new_node

    return [convert_(tree)]


def extract_node_summaries(tree: Dict[str, Any]) -> List[str]:
    """
    Extract the node summaries from the tree.

    :param tree: The tree
    :return: The node summaries
    """
    def extract_(node):
        summaries.append(node['Summary'])
        if 'Children' in node:
            for child in node['Children']:
                extract_(child)

    summaries = []
    extract_(tree)
    return summaries


def extract_label_summaries(tree: Dict[str, Any]) -> List[str]:
    """
    Extract the label summaries from the tree.

    :param tree: The tree
    :return: The label summaries
    """
    def extract_(node):
        label_summary[node['Label']] = node['Summary']
        if 'Children' in node:
            for child in node['Children']:
                extract_(child)

    label_summary = {}
    extract_(tree)
    return label_summary


def extract_terminal_summaries(tree: Dict[str, Any]) -> Dict:
    """
    Extract summaries from terminal nodes of the tree. 

    :param tree: The tree
    :return: The label summaries of terminal nodes
    """
    def extract_(node):
        if 'Children' in node:
            for child in node['Children']:
                extract_(child)
        else:
            label_summary[node['Label']] = node['Summary']

    label_summary = {}
    extract_(tree)
    return label_summary


def stringify_label_summaries(label_summaries):
    """
    Convert the label summaries into a list of strings.

    :param label_summaries: A dictionary of label summaries.
    :return: A list of strings.
    """
    return [f'{label}: {summary}'
            for label, summary in label_summaries.items()]


def extract_node_labels(tree: Dict[str, Any]) -> List[str]:
    """
    Extract the node labels from the tree.

    :param tree: The theme tree
    :return: The node labels
    """

    sums = extract_label_summaries(tree)
    sums = stringify_label_summaries(sums)

    # Remove the top level node
    sums = sums[1:]
    sums = [res.split(':')[0] for res in sums]

    return sums

def extract_terminal_labels(tree: Dict[str, Any]) -> List[str]:
    """
    Extract the terminal labels from the tree.

    :param tree: The theme tree
    :return: The terminal node labels
    """

    sums = extract_terminal_summaries(tree)
    sums = stringify_label_summaries(sums)

    # Remove the top level node
    sums = [res.split(':')[0] for res in sums]

    return sums

def print_tree(node, prefix=''):
    """
    Print the tree.

    :param node: The node
    :param prefix: The prefix
    :return:
    """
    has_children = 'Children' in node and len(node['Children']) > 0

    print(prefix + node['Label'])

    if not has_children:
        return

    for i, child in enumerate(node['Children']):
        is_last = i == (len(node['Children']) - 1)
        if is_last:
            branch = '└── '
            child_prefix = prefix + '    '
        else:
            branch = '├── '
            child_prefix = prefix + '│   '

        print(prefix + branch, end='')
        print_tree(child, child_prefix)


def visualize_tree(tree: Dict[str, Any]) -> None:
    """
    Visualize the tree.

    :param tree: The tree
    :return:
    """
    def extract_labels(node, parent_label=''):
        labels.append(node['Label'])
        parents.append(parent_label)
        if 'Children' in node:
            for child in node['Children']:
                extract_labels(child, node['Label'])

    labels = []
    parents = []
    extract_labels(tree)

    df = pd.DataFrame({'labels': labels,
                       'parents': parents})
    fig = px.treemap(df, names='labels', parents='parents')
    fig.show()

