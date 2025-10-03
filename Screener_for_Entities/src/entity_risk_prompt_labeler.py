from typing import Any, Dict
from bigdata_research_tools.prompts.risk import compose_risk_system_prompt_focus
import ast
from bigdata_research_tools.themes import ThemeTree, themes_default_llm_model_config
from bigdata_research_tools.llm import LLMEngine
from json_repair import repair_json
from typing import List, Optional
import pandas as pd
import os
from .search_entities import post_process_dataframe
from bigdata_research_tools.labeler.risk_labeler import RiskLabeler, map_risk_category

def generate_risk_tree(
    main_theme: str,
    analyst_focus: str = "",
    llm_model_config: Dict[str, Any] = None,
    system_prompt: str = None,
) -> ThemeTree:
    """
    Generate a `ThemeTree` class from a main theme and analyst focus.

    Args:
        main_theme (str): The primary theme to analyze.
        focus (str, optional): Specific aspect(s) to guide sub-theme generation.
            If provided, a two-step process is used to better integrate the focus.
        llm_model_config (dict): Configuration for the large language model used to generate themes.
            Expected keys:
            - `provider` (str): The model provider (e.g., 'openai').
            - `model` (str): The model name (e.g., 'gpt-4o-mini').
            - `kwargs` (dict): Additional parameters for model execution, such as:
            - `temperature` (float)
            - `top_p` (float)
            - `frequency_penalty` (float)
            - `presence_penalty` (float)
            - `seed` (int)

    Returns:
        ThemeTree: The generated theme tree.
    """
    ll_model_config = llm_model_config or themes_default_llm_model_config
    model_str = f"{ll_model_config['provider']}::{ll_model_config['model']}"
    llm = LLMEngine(model=model_str)

    system_prompt = system_prompt or compose_risk_system_prompt_focus(main_theme, focus=analyst_focus)

    tree_str = llm.get_response(
        [{"role": "user", "content": system_prompt}],
        **ll_model_config.get("kwargs", {}),
    )

    tree_str = repair_json(tree_str)

    tree_dict = ast.literal_eval(tree_str)

    return ThemeTree.from_dict(tree_dict)


def add_prompt_fields(df_sentences: pd.DataFrame, additional_prompt_fields: Optional[List] = None
    ) -> List[Dict]:
        """
        Add additional fields from the DataFrame for the labeling prompt.

        Args:
            df_sentences (DataFrame): The DataFrame containing the search results.
            additional_prompt_fields (Optional[List]): Additional fields to be used in the labeling prompt.

        Returns:
            List[Dict]: A list of dictionaries with the additional fields for each row in the DataFrame.
        """
        if additional_prompt_fields:
            missing = set(additional_prompt_fields) - set(df_sentences.columns)
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            else:
                return df_sentences[additional_prompt_fields].to_dict(orient="records")
        else:
            return []

def label_search_results(
        df_sentences: pd.DataFrame,
        terminal_labels : List[str],
        risk_tree: ThemeTree,
        labeler: RiskLabeler,
        additional_prompt_fields: Optional[List] = None,
        main_theme: str = '',
        
    ):
        """
        Label the search results with our theme labels.

        Args:
            df_sentences (DataFrame): The DataFrame containing the search results.
            terminal_labels (List[str]): The terminal labels for the risk categories.
            risk_tree (ThemeTree): The ThemeTree object containing the risk taxonomy.
            prompt_fields (Dict): Additional fields to be used in the labeling prompt.

        Returns:
            DataFrame: The port-processed DataFrame with labeled search results.
        """

        prompt_fields = add_prompt_fields(df_sentences, additional_prompt_fields)

        df_labels = labeler.get_labels(
            main_theme=main_theme,
            labels=terminal_labels,
            texts=df_sentences["masked_text"].tolist(),
            textsconfig=prompt_fields,
        )

        # Merge and process results
        df = pd.merge(df_sentences, df_labels, left_index=True, right_index=True)

        # Create the reverse mapping
        label_to_parent = risk_tree.get_label_to_parent_mapping()

        ## to do: generalize the mapping function to allow for different risk categories and assign it to a class variable

        ## to do: generalize the extra fields generation logic to allow for different fields to be added

        df["risk_factor"] = df["label"].apply(
            lambda x: map_risk_category(x, label_to_parent)
        )

        df = df.loc[
            df.risk_factor.notnull() | df.risk_factor.ne("Not Applicable")
        ].copy()

        df["channel"] = df.apply(
            lambda row: row["risk_factor"] + "/" + row["label"], axis=1
        )

        df["theme"] = main_theme

        df_clean = post_process_dataframe(
            df,
            extra_fields={
                "channel": "Risk Channel",
                "risk_factor": "Risk Factor",
                "quotes": "Highlights",
            },
            extra_columns=["Risk Channel", "Risk Factor", "Highlights"],
        )

        return df, df_clean

from string import Template

def get_risk_entity_labeler_prompt(main_theme: str, label_summaries: list) -> str:
    """
    Generate the risk entity labeler prompt with the provided parameters.
    
    Args:
        main_theme (str): The main risk theme being analyzed
        label_summaries (list): List of risk sub-scenario summaries
        
    Returns:
        str: The formatted prompt string
    """
    return f"""Forget all previous prompts.

You are assisting a professional analyst in evaluating both the exposure and risk classification for "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" regarding the Risk Scenario "{main_theme}".
This involves a two-step process: confirming exposure of "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" and classifying specific risks if exposure is confirmed. Use the headline for contextual understanding.

<input_details>
You will receive the following information::
- ID: [text ID]
- Headline: [The Headline of the News Article containing Text]
- Text: [Paragraph requiring analysis]
- Risk Scenario: "{main_theme}"
</input_details>

Follow these guidelines:

<exposure_assessment>
- Examine whether the text explicitly mentions the Risk Scenario "{main_theme}" or any of its core components.
- Ensure that "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" is the main focus of the text and that it is clearly stated that "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" is facing or will face consequences caused by the Risk Scenario "{main_theme}".
- Assess if there are DIRECT consequences on "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}’s" internal and external activities, operations, future performance, stability, sustainability, and long term growth.
- Designate the exposure as unclear if the text lacks an explicit DIRECT link between "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" and the Risk Scenario
- Designate the exposure as unclear if the text relies on generic information.
</exposure_assessment>

<risk_classification>
If direct exposure of {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} is confirmed:

- Identify and classify the specific risk using this list of Risk Sub-Scenarios:
    "{label_summaries}".

- Follow a detailed classification process:
    - Examine the text to confirm how the Risk Scenario "{main_theme}" directly impacts "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" through one of the Risk Sub-Scenarios.
    - Write a concise motivation that explains the direct link between "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" and the Risk Sub-Scenario as stated in the text.
    - The motivation should always start with "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}".
    - Identify an appropriate Risk Sub-Scenario label from the list that describes explicitly the impact on {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s internal and external activities, operations, stability, sustainability, and long term growth or performance.
    - Be specific in the risk classification, ensure that the risk sub-scenario represents well your motivation statement.
    - Ensure that the Risk Sub-Scenario label can be directly extracted from the text that it describes with high granularity how "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" is affected.
    - Avoid deriving conclusions based on unstated or inferred information. Focus only on the explicit content of the text or headline.
</risk_classification>

<verbatim_quotes_extraction>
- Extract verbatim quotes from the text that support the classification and illustrate {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s exposure to the specific Risk Sub-Scenario.
- Ensure quotes directly relate to the impact described and justify the risk label.
- Extract full sentences or phrases that clearly indicate, as standalone statements, how "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" is affected by the Risk Scenario "{main_theme}" and the Sub-Scenario label assigned.
</verbatim_quotes_extraction>

<response_format>
Structure your response as a JSON object containing:
"sentence_id": "<sentence_id>"
"motivation": : A concise explanation describing the link between "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}" and the Risk Sub-Scenario.
"label": State the specific risk Sub-Scenario label or 'unclear'.
"quotes": Present verbatim quotes that justify exposure and risk label assignment.

{{"<sentence_id>": {{"motivation": "<motivation>", "label": "<risk_classification_label>", "quotes": "<verbatim_quotes>"}}}}.
</response_format>

<examples>
ID: 3
Headline: "Tariffs to Strain Supply Chains Globally"
Text: "New tariffs against China will significantly impact {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s operations due to its reliance on raw materials from Chinese suppliers."
Scenario: "New Tariffs against China"
Output:

{{3:{{
  "motivation": "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s supply operations are directly impacted by new tariffs due to their reliance on raw materials sourced from China.",
  "label": "Supply Chain Disruption",
  "quotes": ["New tariffs against China will significantly impact {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s operations", "reliance on raw materials from Chinese suppliers"]}}
}}

ID: 5
Headline: "Interest Rate Fluctuations to Affect Markets"
Text: "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s analysts are forecasting higher risks associated with potential interest rate changes."
Scenario: "Interest Rate Volatility"
Output:

{{5:{{
  "motivation": "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s analysts are forecasting higher risks associated with potential interest rate changes.",
  "label": "unclear",
  "quotes": []
}}}}

ID: 2
Headline: "Economic Challenges Ahead Due to Tariffs on China"
Text: "{os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')}'s analysts report a potential economic downturn in {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} linked to new tariffs against China."
Risk Scenario: "New Tariffs Against China"
Output:

{{2:{{
  "motivation": "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}'s analysts are assessing the potential economic impact of new tariffs against China.",
  "label": "Economic Downturns",
  "quotes": ["{os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')}'s analysts report a potential economic downturn in {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}"]}}
}}

ID: 3
Headline: "Analyzing External Factors in Business Strategy"
Text: "{os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')} is studying external factors such as tariffs to gauge potential risks in {os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')}'s analysts report a potential economic downturn in {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}."
Risk Scenario: "New Tariffs on Semiconductors"
Output:

{{3:{{
  "motivation": "{os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')}'s analysts are studying external factors such as tariffs to gauge potential risks in {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}."
  "label": "unclear",
  "quotes": []}}
}}

ID: 4
Headline: "Market Trends Influence Stock Performance"
Text: "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}’s stock is influenced by broad market trends."
Risk Scenario: "Increased Uncertainty and Volatility"
Output:

{{4:{{
  "motivation": "The text does not related to the Risk Scenario and it does not mention any specific risk sub-scenario affecting {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}.",
  "label": "unclear",
  "quotes": []}}
}}

ID: 5

Headline: "Tariffs and Their Economic Impact"
Text: "{os.environ.get('BIGDATA_OTHER_ENTITY_PLACEHOLDER')} researchers estimate that tariffs will affect the broader economy in {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')}."
Risk Scenario: "New Tariffs against China"
Output:

{{5:{{
  "motivation": "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} is not linked with any specific risk sub-scenario or any tangible effect of the Risk Scenario.",
  "label": "unclear",
  "quotes": []}}
}}

ID: 2
Headline: "China Tariffs Impact Supply Chains"
Text: "According to recent reports, {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} is heavily dependent on China. The recent tariffs against China have forced {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} to reconsider its supply chain, potentially leading to increased logistics costs."
Risk Scenario: "New Tariffs against China"
Output:

{{2:{{
  "motivation": "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} is said to be reconsidering its supply chain in the face of the risk scenario. The text clearly links {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} with the Risk Scenario and mentions an explicit Sub-scenario risk of Supply Chain Disruptions.",
  "label": "Supply Chain Disruption",
  "quotes": [
    "{os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} is heavily dependent on China",
    "The recent tariffs against China have forced {os.environ.get('BIGDATA_TARGET_ENTITY_PLACEHOLDER')} to reconsider its supply chain, potentially leading to increased logistics costs."
  ]}}
}}
</examples>

"""


def get_entity_risk_prompt(main_theme: str, analyst_focus: str = "") -> str:
    """
    Generate the entity risk prompt with the provided parameters.
    
    Args:
        main_theme (str): The main risk theme being analyzed
        analyst_focus (str): Specific analyst focus area
        
    Returns:
        str: The formatted prompt string
    """
    return f"""
            Forget all previous prompts.
            You are assisting a professional risk analyst tasked with creating a taxonomy to classify the impact of the Risk Scenario '**{main_theme}**' on other entities, specifically countries, geographical places, and organizations.
            Your objective is to generate a **comprehensive tree structure** that maps the **risk spillovers** stemming from the Risk Scenario '**{main_theme}**', and generates related sub-scenarios. 

            Key Instructions:

            1. **Understand the Risk Scenario: '{main_theme}'**:
                - The Risk Scenario '**{main_theme}**' represents a central, multifaceted concept that may be harmful or beneficial to an entity.
                - Your task is to identify how the Risk Scenario impacts entities through various **risk spillovers** and transmission channels.
                - Summarize the Risk Scenario '**{main_theme}**' in a **short list of essential keywords**.
                - The keyword list should be short (1-2 keywords). Avoid unnecessary, unmentioned, indirectly inferred, or redundant keywords.

            2. **Create a Tree Structure for Risk Spillovers and Sub-Scenarios**:
                - Decompose the Risk Scenario into **distinct, focused, and self-contained risk spillovers**.
                - Each risk spillover must represent a **specific risk channel** through which entities are exposed to as a consequence of the Risk Scenario.
                - Label each **primary node** in the tree explicitly as a "Risk" in the `Label` field. For example:
                    - Use "Cost Risk" instead of "Cost Impacts."
                    - Use "Supply Chain Risk" instead of "Supply Chain Disruptions."
                - Risk spillovers must:
                    - Cover a wide range of potential impacts on entities' and long-term stabiliity and growth.
                    - Explore both macroeconomic and microeconomic dimensions of the Risk Scenario '**{main_theme}**' and analyze their impact on entities when relevant.
                    - Include **direct and indirect consequences** of the main scenario.
                    - Represent **dimensions of risk** that entities must monitor or mitigate.
                    - NOT overlap.
                - Independently identify the most relevant spillovers based on the Risk Scenario '**{main_theme}**', without limiting to predefined categories.

            3. **Generate Sub-Scenarios for Each Risk Spillover**:
                - For each risk spillover, identify **specific sub-scenarios** that will arise as a consequence of the Risk Scenario '**{main_theme}**'.
                - All sub-scenarios must:
                    - Be **concise and descriptive sentences**, clearly stating how the sub-scenario is an event caused by the main scenario.
                    - **Explicitly include ALL core concepts and keywords** from the main scenario, including specific geographical locations or temporal details, in every sentence in order to ensure clarity and relevance towards the main scenario.
                    - Integrate the Risk Scenario in a natural way, avoiding repetitive or mechanical structures.
                    - Not exceed 15 words.
                - Sub-scenarios MUST be mutually exclusive: they CANNOT overlap neither within nor across branches of the tree.
                - Do NOT combine multiple sub-scenarios in a single label.
                - Sub-Scenarios have to be consistent with the parent Risk Spillover (e.g. Market Access related sub-scenarios have to belong to the Market Access Risk node).
                - Generate 3 OR MORE sub-scenarios for each risk spillover.
                - Generate a short label for each subscenario.

            4. **Iterate Based on the Analyst's Focus: '{analyst_focus}'**:
                - After generating the initial tree structure, use the analyst's focus ('{analyst_focus}') to:
                    - Identify **missing branches** or underexplored areas of the tree.
                    - Add new risk spillovers or sub-scenarios that align with the analyst's focus.
                    - Ensure that sub-scenarios ALWAYS include ALL core components of the Risk Scenario and are formulated as natural sentences.
                    - Ensure that sub-scenarios DO NOT overlap within and across risk spillovers.
                    - Ensure that sub-scenarios belong to the correct Risk Spillover.
                - If the analyst focus is empty, skip this step.
                - If you don't understand the analyst focus ('{analyst_focus}'), ask an open-ended question to the analyst.

            5. **Review and Expand the Tree for Missing Risks**:
                - After incorporating the analyst's focus, review the tree structure to ensure it includes a **broad range of risks** and sub-scenarios.
                - Add any missing risks or sub-scenarios to the tree.

            6. **Format Your Response as a JSON Object**:
                - Each node in the JSON object must include:
                    - `Node`: an integer representing the unique identifier for the node,
                    - `Label`: a string for the name of the risk factor or the sub-scenario label.
                    - `Summary`: a short sentence describing the sub-scenario,
                    - `Children`: an array of child nodes.
                    - For the Risk Scenario, include a list of core concepts in the field `Keywords`. 
        ### Example Structure:
        **Risk Scenario: Global Warming effects in the United States**
        {{
            "Node": 1,
            "Label": "Global Warming effects in the United States",
            "Children": [
            {{
                "Node": 2,
                "Label": "Supply Chain Risk",
                "Children": [
                {{
                    "Node": 3,
                    "Label": "Transportation Delays",
                    "Summary": "Firms in the United Stated will be experiencing supply chain disruptions and transportation delays due to extreme weather",
                    "Children": []
                }},
                {{
                    "Node": 4,
                    "Label": "Lower Agricultural Yields",
                    "Summary":"Global warming will reduce agricultural yields in the United States, damaging firms' supply chains",
                    "Children": []
                }}
                ]
            }}
            ],
            "Keywords": ["Global Warming", "United States"]
        }}
        **Risk Scenario: Tariffs against China affect US companies**
        {{
            "Node": 1,
            "Label": "Tariffs against China affect US companies",
            "Children": [
            {{
                "Node": 2,
                "Label": "Trade Related Risks",
                "Children": [
                {{
                    "Node": 3,
                    "Label": "Retaliatory Tariffs",
                    "Summary": "New tariffs could trigger retaliatory tariffs from China",
                    "Children": []
                }},
                {{
                    "Node": 4,
                    "Label": "US Dollar / Chinese Yuan Currency Volatility",
                    "Summary":"US faces uncertainty in trade as new tariffs will increase volatility in the USD/CNY.",
                    "Children": []
                }}
                ]
            }}
            ],
            "Keywords": ["Tariffs", "China"]
        }}
        """