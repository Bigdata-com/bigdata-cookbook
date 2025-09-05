"""
Copyright (C) 2024, RavenPack | Bigdata.com. All rights reserved.
Author: Alessandro Bouchs (abouchs@ravenpack.com)
"""


from bigdata_client import Bigdata
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_client.query import Keyword, Entity, Any
from bigdata_client.daterange import AbsoluteDateRange

import graphviz
import time
from tqdm.notebook import tqdm
import pandas as pd
import hashlib
import plotly.express as px
from IPython.display import display, HTML
import re
import openai
import html
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from collections import defaultdict
MODEL_NAME = 'gpt-4o-mini'
import ast
# Function to add line breaks
def add_line_breaks(text, max_len=50):
    
    if isinstance(text, list):
        text = "; ".join(text)
    
    if type(text)==str:
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > max_len:
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " "
                current_line += word
        lines.append(current_line)
        return "<br>".join(lines)
    else:
        return None
    
def extract_entity_name(entity_key, mapping):
    # Search the dictionary for the matching key
    for name, key in mapping.items():
        if key == entity_key:
            return name
    return None  # Return None if key is not found

def drop_duplicates(df = pd.DataFrame(), date_column = '', duplicate_set = []):
    df = df.sort_values(by=date_column)
    non_dup_index = []
    dup_index = []
    for i, gp in df.groupby(duplicate_set):
        non_dup_index.append(gp.iloc[0].name)
        dup_index.extend(gp.iloc[1:].index.values)
    deduplicated_df = df.loc[non_dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    duplicates = df.loc[dup_index,:].copy().sort_values(by=date_column).reset_index(drop=True)
    
    return deduplicated_df, duplicates

def simplify_tree(tree):
    """
    Simplifies a tree structure by keeping only 'Label' and 'Children' fields.

    Args:
        tree (dict): The input tree structure.

    Returns:
        dict: A simplified tree structure with only 'Label' and 'Children'.
    """
    simplified = {
        'Label': tree['Label'],
        'Children': [simplify_tree(child) for child in tree.get('Children', [])]
    }
    return simplified

def tree_to_string(tree, level=0):
    """
    Converts a tree structure to a string with indented hierarchy.

    Args:
        tree (dict): The input tree structure with 'Label' and 'Children'.
        level (int): The current indentation level.

    Returns:
        str: A string representing the tree with indents for hierarchy.
    """
    indent = "  " * level
    tree_str = f"{indent}- {tree['Label']}\n"
    for child in tree.get('Children', []):
        tree_str += tree_to_string(child, level + 1)
    return tree_str

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

from typing import Dict, Any, List

def generate_theme_taxonomy(openai_api_key, 
                            main_theme,
                            analyst_focus):

    """
    Generates a nested tree structure dictionary of sub-themes related to a theme.
        main_theme: Main topic.
        analyst_focus: Focus from the human analyst.
    """

	# Prompt instructed to ChatGpt to return the structure tree as a dictionary.
    system_prompt = f"""
    Forget all previous prompts.
    You are assisting a professional macroeconomic analyst tasked with creating a structured taxonomy to analyze the macro theme '**{main_theme}**' and its implications for companies.
    Your objective is to generate a **comprehensive tree structure** that maps the relationships between the theme '**{main_theme}**', its components, and the drivers associated with it.

    Key Instructions:

    1. **Understand the Macro Theme: '{main_theme}'**:
       - The theme '**{main_theme}**' represents a complex economic phenomenon that can impact firms in various ways.
       - Summarize the theme '**{main_theme}**' in a **short list of essential keywords** (1-2 keywords).

    2. **Identify Components and Drivers**:
       - Based on the theme '**{main_theme}**', identify **relevant components** that are affected by it. Components may include various sectors, industries, or economic factors.
       - For each identified component, determine the **drivers** that influence it. Drivers may include economic forces, policy changes, or market dynamics.
       - Reflect on the components and what they mean in macroeconomic terms without speculating. If Food is a main component of Inflation, then your main drivers will focus on the costs that the agricultural sector is facing, the trade agreements that can affect the market, etc.
       - Likewise, if some components do not bear any relationship with a suggested driver, then do not include it in the branch. For instance, if monetary policy is not expected to affect food prices, then it is not a realistic driver of that component.
       - When creating a driver, think about why and how that driver would actually affect '**{main_theme}**' through that component. Assess cause-effect relationships like an economist. 
       
    3. **Analyst Focus**:
        - {analyst_focus}
        - Use the analyst focus to steer your identification of components and drivers, ensuring they are relevant to the analysis.

    4. **Construct the Tree Structure**:
       - Create a hierarchical tree structure that represents the relationships as follows:
         - The root node should be the macro theme '**{main_theme}**'.
         - Each child node should represent a **component** related to the theme.
         - Under each component, include child nodes for the **drivers** that influence that component.
         - The tree should end with the drivers, meaning there should be no further children under the drivers.

    5. **Generate a Summary Sentence for Each Driver**:
       - For each driver under a component, write a **specific sentence** that illustrate how the driver affects the theme '**{main_theme}**' through that component.
       - Sub-scenarios must be concise, descriptive sentences (not exceeding 15 words) that clearly state the relationship to the main theme and the specific driver. They also include all distinguishing terms for the main theme, including specific geographical places and countries.
       - The sentences should be realistic sentences expressed in news and traditional media. A generic sentence like "High Food Prices lead to increase inflation in the United States" is not realistic and not useful.
       - Generate **3 or more sub-scenarios** for each driver, ensuring they are mutually exclusive.

    6. **Iterate Based on the Analyst's Focus: '{analyst_focus}'**:
        - After generating the initial tree structure, use the analyst's focus ('{analyst_focus}') to identify any **missing components** or underexplored drivers.
        - Add new components, drivers, or sub-scenarios that align with the analyst's focus, ensuring clarity and relevance.

    7. **Format Your Response as a JSON Object**:
        - Each node in the JSON object must include:
            - `Node`: an integer representing the unique identifier for the node,
            - `Label`: a string for the name of the component or driver,
            - `Summary`: a short sentence describing the component or driver,
            - `Children`: an array of child nodes (which will be empty for drivers),
            - For the main theme, include a list of core concepts in the field `Keywords`.
""" + """
    ### Example Structure:
    **Main Theme: United States Inflation**
    {
  "Node": 1,
  "Label": "United States Inflation",
  "Summary": "Inflation in the United States.",
  "Keywords": ["United States", "inflation", "price increase", "price decrease"],
  "Children": [
    {
      "Node": 2,
      "Label": "Food",
      "Summary": "Inflation in the United States is driven by increasing food prices.",
      "Children": [
        {
          "Node": 3,
          "Label": "Consumer Demand",
          "Summary": "Increased consumer demand in the United States drives food prices higher.",
          "Children": []
        },
        {
          "Node": 4,
          "Label": "Labour Costs",
          "Summary": "Rising labour costs lead to higher food prices in the US.",
          "Children": []
        }
      ]
    },
    {
      "Node": 5,
      "Label": "Energy",
      "Summary": "Energy has been a main component of recent inflation bursts in the United States.",
      "Children": [
        {
          "Node": 6,
          "Label": "Energy Consumption",
          "Summary": "Increased energy consumption in the United States drives prices higher.",
          "Children": []
        },
        {
          "Node": 7,
          "Label": "Production Costs",
          "Summary": "Increased production costs lead to higher energy prices in the US.",
          "Children": []
        }
      ]
    },
    {
      "Node": 8,
      "Label": "Wages",
      "Summary": "Rising wages in the United States are a response to inflationary pressures and a tightening labor market.",
      "Children": [
        {
          "Node": 9,
          "Label": "Skilled Labour",
          "Summary": "In sectors with high demand for skilled labor, wage increases occur as companies compete for talent, contributing to overall inflation.",
          "Children": []
        },
        {
          "Node": 10,
          "Label": "Cost Transfer",
          "Summary": "Higher wages lead to increased production costs for businesses, which are passed on to consumers, further fueling inflation.",
          "Children": []
        }
      ]
    }
  ]
}
    """
    #print(system_prompt)

    # Initialize OpenAI client
    client = openai.OpenAI(api_key = openai_api_key)
    
    response = client.chat.completions.create(
      model = "gpt-4o-mini",
      messages = [
        {
          "role": "system",
          "content": system_prompt
        },
        # {
        #   "role": "user",
        #   "content": main_theme
        # },
        # {
        #   "role": "user",
        #   "content": analyst_focus
        # }
      ],
      temperature = 0, # we want a small temperature
      top_p = 1,
      frequency_penalty = 0,
      presence_penalty = 0,
      seed = 123,
      response_format = {"type": "json_object"}
    )
    
    tree_string = response.model_dump()['choices'][0]['message']['content'] 
    tree_string = re.sub('```', '', tree_string)
    tree_string = re.sub('json', '', tree_string)

    # Convert string into dictionary
    tree_dictionary = ast.literal_eval(tree_string)

    return tree_dictionary,tree_dictionary.get("Keywords", []) 


def extract_node_labels(tree_dictionary):
    """
    Extracts the array of node labels from a nested tree structure.
    Parameters:
        tree_dictionary (dict): The tree structure represented as a dictionary.
    Returns:
        list: A list of all node labels in the tree.
    """
    nodes_labels = []

    # Helper function to recursively extract labels
    def traverse(node):
        # Add the label of the current node
        nodes_labels.append(node.get('Summary','Label'))
        
        # If the node has children, traverse them
        if 'Children' in node:
            for child in node['Children']:
                traverse(child)

    # Start traversing from the root node
    traverse(tree_dictionary)
    
    return nodes_labels


def extract_node_summaries(tree_dictionary):
    """
    Extracts the list of summaries related to the nodes from a nested tree structure.
    Parameters:
        tree_dictionary (dict): The tree structure represented as a dictionary.
    Returns:
        list: A list of all summaries in the tree.
    """
    nodes_summaries = []
    
    if 'Summary' in tree_dictionary:
        nodes_summaries.append(tree_dictionary['Summary'])
        
    if 'Children' in tree_dictionary:
        for child in tree_dictionary['Children']:
            nodes_summaries.extend(extract_node_summaries(child))
            
    return nodes_summaries

def generate_mindmap(tree_dictionary):
    """
    Creates a vertical mind map from the given tree structure.
        tree_dictionary: A nested dictionary representing the tree structure.
    """
    mindmap = graphviz.Digraph()

    # Set direction to left-right
    mindmap.attr(
        rankdir='LR', 
        ordering='in',
        splines='curved',
    ) 

    def add_nodes(node):
        # Add a node to the mind map with a box shape
        mindmap.node(
            str(node["Node"]), 
            node.get("Summary", node.get("Label","")), 
            shape="box", 
            style="filled", 
            fillcolor='lightgrey', 
            margin="0.2,0", align="left", 
            fontsize="12", fontname="Arial",
        )
        
        # If the node has children, recursively add them
        if "Children" in node:
            for child in node["Children"]:
                # Add an edge from the parent to each child
                mindmap.edge(
                    str(node["Node"]), 
                    str(child["Node"]),
                )
                # Recursively add child nodes
                add_nodes(child)

    # Start with the root node
    add_nodes(tree_dictionary)

    # Return the Graphviz dot object for rendering
    return mindmap

def generate_themes_mindmap_agent(openai_api_key, 
                                  main_theme,
                                  analyst_focus):

    tree_dictionary, theme_keywords = \
        generate_theme_taxonomy(openai_api_key = openai_api_key, 
                                main_theme = main_theme,
                                analyst_focus = analyst_focus)

    nodes_labels = extract_node_labels(tree_dictionary)[1:]

    nodes_summaries = extract_node_summaries(tree_dictionary)[1:]
    
    mindmap = generate_mindmap(tree_dictionary)

    return(tree_dictionary,theme_keywords, nodes_labels, nodes_summaries, mindmap)
    
def string_to_dict(tree_string):
    """
    Converts an indented tree string into a nested dictionary, dropping the top-level root node.

    Args:
        tree_string (str): The input indented string.

    Returns:
        dict: A nested dictionary representing the tree structure.
    """
    lines = tree_string.splitlines()
    stack = []
    root = {}

    for line in lines:
        if not line.strip():  # Skip empty lines
            continue

        # Calculate indentation level
        stripped_line = line.strip()
        level = (len(line) - len(stripped_line)) // 2  # Assuming two spaces for each level

        # Prepare a new node
        node = stripped_line.lstrip('- ').strip()

        # Adjust stack to current level
        while len(stack) > level:
            stack.pop()

        if level == 0:
            # Skip the top-level node
            if not stack:
                stack.append(root)  # Set the root as the stack base
                continue

        # Add the node to the tree
        current = stack[-1]
        if isinstance(current, dict):
            current[node] = {}
            stack.append(current[node])

    # Remove the empty top-level key
    return root if len(root) > 1 else list(root.values())[0]

def aggregate_column(x, column_name):
    if column_name == 'key_drivers':  # Customize behavior for specific columns
        return "; ".join(filter(None, x[column_name].astype(str).unique()))
    elif 'novelty' in column_name:
        return ", ".join(map(str, filter(None, x[column_name].astype(str).unique())))
    else:
        return ", ".join(filter(None, x[column_name].astype(str).unique()))



import html
import networkx as nx
import plotly.graph_objects as go

    
def truncate_text(text, max_length=300):
    """
    Truncate the given text to a specified length and add an ellipsis if needed.
    
    Args:
        text (str): The text to truncate.
        max_length (int): The maximum length of the text.
    
    Returns:
        str: The truncated text.
    """
    if len(text) > max_length:
        return text[:max_length] + "... [truncated]"
    return text

def get_system_prompt(theme_tree, main_theme):
    label_summaries = extract_label_summaries(theme_tree)
    label_summaries = stringify_label_summaries(label_summaries)
    label_summaries = label_summaries[1:]  # Remove the root - AI Solutions
    return compose_labeling_system_prompt(main_theme, label_summaries)

def extract_label_summaries(tree: Dict[str, Any]) -> List[str]:
    def extract_(node):
        label_summary[node['Label']] = node['Summary']
        if 'Children' in node:
            for child in node['Children']:
                extract_(child)

    label_summary = {}
    extract_(tree)
    return label_summary

def stringify_label_summaries(label_summaries):
    return [f'{label}: {summary}'
            for label, summary in label_summaries.items()]

def calculate_number_of_hits(
    df):
    """
    Process the DataFrame to prepare data for company-risk network visualization,
    with filtering based on selected labels.

    Args:
        df (pd.DataFrame): Input dataframe containing risk factors and related metadata.
        labels (list): List of valid direction labels to include in the analysis (e.g., ['P', 'N', 'U']).
        consolidated_summaries (bool): Flag to determine if summaries should be consolidated.
        consolidate_impact_prompt (str): Prompt template for consolidating impacts.
        theme (str): The topic/theme related to the risk.

    Returns:
        pd.DataFrame: Processed dataframe with risk-company links, counts, KPIs, impacts, concatenated content, and summaries.
    """

    # Group by company and risk channel
    grouping_columns = [
        'entity_name', 'rp_entity_id', 'entity_sector', 
        'entity_industry','entity_ticker','channel','risk_factor','sub_scenario', 
    ]
    
    aggregations = {
        'hit_count': ('sentence_id', 'nunique'),
        'date': ('date', lambda x: list(x.dropna())),
        'headline': ('headline', lambda x: list(x.dropna())),
        'text': ('text', lambda x: list(x.dropna())),
        'source': ('source_name', lambda x: list(x)),
        'quotes': ('quotes', lambda x: list(x.dropna()))
    }

    # Perform the grouping and aggregation
    df_grouped = df.groupby(grouping_columns).agg(
        **aggregations
    ).reset_index()
    
    # Create a single concatenated_content column by combining all relevant texts
    def concatenate_content(row):
        combined = []
        for date,channel, headline, quotes, text, source in zip(
            row['date'],row['channel'], row['headline'], row['quotes'], row['text'], row['source']
        ):
            quotes_formatted = [f'"{quote}"' for quote in quotes]
            combined.append(f"Channel: {row['channel']}\n\n"
                f"Date: {date}\n\n"
                # f"Headline: {headline}\n\n"
                # f"Text: {text}\n\n"
                f"Quotes: {', '.join(quotes_formatted)}\n\n"
                f"Source: {source}\n\n"
            )
        return "\n\n".join(combined) if combined else None

    df_grouped['concatenated_content'] = df_grouped.apply(concatenate_content, axis=1)

    df_risk_links = df_grouped.sort_values('hit_count', ascending=False).reset_index(drop=True)

    return df_risk_links


def generate_pivot_table(df, value = 'hit_count', fields = ['entity_name','entity_sector', 'entity_ticker'], columns = ['sub_scenario']):
    hits_df = df.pivot_table(values=value, index = fields, columns = columns, aggfunc='sum', fill_value=0)
    
    hits_df['Composite Score'] = hits_df.sum(axis=1)
    hits_df = hits_df.reset_index()
    hits_df.columns.name = None
    hits_df.index.name = None
    hits_df = hits_df.sort_values(by='Composite Score',
                                    ascending=False).reset_index(drop=True)
    
    def highlight_heatmap(df):
        """
        Apply a color gradient to the DataFrame to create a heatmap-like effect.
        """
        return df.style.background_gradient(cmap='RdYlGn_r', axis=None)  # Use reversed RdYlGn for green to red

    # Apply the heatmap-style formatting
    # styled_df = highlight_heatmap(hits_df)

    return hits_df

def format_sub_scenario_labels(tree_dict):
    ## Use only the summary of the leaf nodes
    node_labels = get_most_granular_elements(tree_dict, 'Label')
    sub_scenario_summaries = get_most_granular_elements(tree_dict, 'Summary')
    sub_scenarios_list = [f'{x}: {y}' for x,y in zip(node_labels,sub_scenario_summaries)]
    return sub_scenarios_list

def clean_dataframe(df):
    return df[['timestamp_utc','date', 'rp_document_id', 'sentence_id', 'headline',
       'rp_entity_id', 'entity_name', 'entity_sector', 'entity_industry', 'entity_ticker', 'text','source_name', 'source_rank', 'url','risk_factor', 'sub_scenario', 'channel', 'quotes']].copy().sort_values('timestamp_utc', ascending=True).reset_index(drop=True)