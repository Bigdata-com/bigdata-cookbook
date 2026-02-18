from os import environ


def get_other_entity_placeholder() -> str:
    return environ.get("BIGDATA_OTHER_ENTITY_PLACEHOLDER", "Other Company")


def get_target_entity_placeholder() -> str:
    return environ.get("BIGDATA_TARGET_ENTITY_PLACEHOLDER", "Target Company")


narrative_system_prompt_template: str = """
Forget all previous prompts.
You are assisting in tracking narrative development within a specific theme. 
Your task is to analyze sentences and identify how they contribute to key narratives defined in the '{theme_labels}' list.

Please adhere to the following guidelines:

1. **Analyze the Sentence**:
   - Each input consists of a sentence ID and the sentence text
   - Analyze the sentence to determine if it clearly relates to any of the themes in '{theme_labels}'
   - Your goal is to select the most appropriate label from '{theme_labels}' that corresponds to the content of the sentence. 
   
2. **Label Assignment**:
   - If the sentence doesn't clearly match any theme in '{theme_labels}', assign the label 'unclear'
   - Evaluate each sentence independently, using only the context within that specific sentence
   - Do not make assumptions beyond what is explicitly stated in the sentence
   - You must not create new labels or choose labels not present in '{theme_labels}'
   - The connection to the chosen narrative must be explicit and clear

3. **Response Format**:
   - Output should be structured as a JSON object with:
     1. A brief motivation for your choice
     2. The assigned label
   - Each entry must start with the sentence ID
   - The motivation should explain why the specific theme was selected based on the sentence content
   - The assigned label should be only the string that precedes the colon in '{theme_labels}'
   - Format your JSON as follows:  {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>"}}, ...}}.
   - Ensure all strings in the JSON are correctly formatted with proper quotes
"""

screener_system_prompt_template: str = """
 Forget all previous prompts.
 You are assisting a professional analyst in evaluating the impact of the theme '{main_theme}' on a company "Target Company".
 Your primary task is first, to ensure that each sentence is explicitly related to '{main_theme}', and second, to accurately associate each given sentence with
 the relevant label contained within the list '{label_summaries}'.

 Please adhere strictly to the following guidelines:

 1. **Analyze the Sentence**:
    - Each input consists of a sentence ID, a company name ('Target Company'), and the sentence text.
    - Analyze the sentence to understand if the content clearly establishes a connection to '{main_theme}'.
    - Your primary goal is to label as '{unknown_label}' the sentences that don't explicitly mention '{main_theme}'.
    - Analyze the list of labels '{label_summaries}' used for label assignment. '{label_summaries}' is a Python list variable containing distinct labels and their definition in format 'Label: Summary', you must pick label only from 'Label' part which means left side of the semicolon for each Label:Summary pair.
    - Your secondary goal is to select the most appropriate label from '{label_summaries}' that corresponds to the content of the sentence.

 2. **First Label Assignment**:
    - Assign the label '{unknown_label}' to the sentence related to "Target Company" when it does not explicitly mentions '{main_theme}'. Otherwise, don't assign a label.
    - Evaluate each sentence independently, focusing solely on the context provided within that specific sentence.
    - Use only the information contained within the sentence for your label assignment.
    - When evaluating the sentence, "Target Company" must clearly mention that its business activities are impacted by '{main_theme}'.
    - Many sentences are only tangentially connected to the topic '{main_theme}'. These sentences must be assigned the label '{unknown_label}'.

 3. **Second Label Assignment**:
    - For the sentences not labeled as '{unknown_label}' and only for them, assign a unique label from the list '{label_summaries}' to the sentence related to "Target Company".
    - Evaluate each sentence independently, focusing solely on the context provided within that specific sentence.
    - Use only the information contained within the sentence for your label assignment.
    - Ensure that the sentence clearly establishes a connection to the label you assigned and to the theme '{main_theme}'.
    - You must not create a new label or choose a label that is not present in '{label_summaries}'.
    - If the sentence does not explicitly mention the label, assign the label '{unknown_label}'.
    - When evaluating the sentence, "Target Company" must clearly mention that its business activities are impacted by the label assigned and '{main_theme}'.

 4. **Response Format**:
    - Your output should be structured as a JSON object that includes:
          1. A brief motivation for your choice.
          2. The assigned label.
          3. The revenue generation.
          4. The cost efficiency.
    - Each entry must start with the sentence ID and contain a clear motivation that begins with "Target Company".
    - The motivation should explain why the label was selected from '{label_summaries}' based on the information in the sentence and in the context of '{main_theme}'. It should also justify the label that had been assigned to the revenue generation and cost efficiency.
    - Ensure that the exact context is understood and labels are based only on explicitly mentioned information in the sentence. Otherwise, assign the label '{unknown_label}'.
    - The assigned label should be only the string that precedes the character ':'.
    - The revenue generation should be either 'Nan' (no mentions), 'low', 'medium' or 'high', and must define whether "Target Company" is generating revenues with the label assigned.
    - The cost efficiency should be either 'Nan' (no mentions), 'low', 'medium' or 'high', and must define to whether "Target Company" is reducing costs with the label assigned.
    - Format your JSON as follows: {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>", "revenue_generation": "<revenue_generation>", "cost_efficiency": "<cost_efficiency>"}}, ...}}.
    - Ensure that all strings in the JSON are correctly formatted with proper quotes.
 """

# Custom screener prompt: main theme must be discussed in relation to the Target Company.
# No revenue_generation/cost_efficiency; output only motivation and label.
screener_theme_target_company_prompt_template: str = """
Forget all previous prompts.
You are assisting a professional analyst in evaluating whether the theme '{main_theme}' is discussed **in relation to "Target Company"**.
Your primary task is, first, to ensure that each sentence  discusses '{main_theme}' **as it relates to or affects "Target Company"** . Second, to assign the most appropriate label from '{label_summaries}' when the theme is connected the Target Company.

Please adhere strictly to the following guidelines:

1. **Analyze the Sentence**:
   - Each input consists of a sentence ID, a company name ("Target Company"), and the sentence text.
   - The main theme '{main_theme}' must be discussed **in connection with "Target Company"**: the sentence must indicate that the theme concerns, affects, applies, relates to the Target Company. If the theme is only discussed in general or in relation to other companies/entities, assign '{unknown_label}'.
   - '{label_summaries}' is a list of labels in format 'Label: Summary'. Use only the 'Label' part (left of the colon) for your assignment.
    - For sentences not labeled '{unknown_label}', assign exactly one label from '{label_summaries}' that best matches how the theme relates to "Target Company".
   - Do not create new labels; use only labels present in '{label_summaries}'.
   - If no label fits the explicit content, assign '{unknown_label}'.

2. **Response Format**:
   - Your output must be a JSON object with:
     1. A brief motivation for your choice (start with "Target Company" when relevant).
     2. The assigned label.
   - Format: {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>"}}, ...}}.
   - Ensure all strings in the JSON are correctly formatted with proper quotes.
"""


def get_screener_theme_target_company_prompt(
    main_theme: str, label_summaries: list[str], unknown_label: str
) -> str:
    """Generate a screener system prompt where the main theme must relate to the Target Company."""
    return screener_theme_target_company_prompt_template.format(
        main_theme=main_theme,
        label_summaries=label_summaries,
        unknown_label=unknown_label,
    )


patent_prompts: dict[str, str] = {
    "filing": """
You are analyzing text to detect patent filing activities by "Target Company". 
Determine if the text describes a legitimate patent filing.

Check for:
1. Explicit mention of new patent filing
2. "Target Company" as the filing entity

Exclude:
- Patent infringement
- Patent expiry
- Filing rejections
- Filing revocations
- Legal issues
- General discussion

Format response as a JSON object with this schema:
{
  "relevant": boolean,
  "explanation": "Brief explanation of classification"
}
""",
    "object": """
Extract and summarize the key patentable innovation mentioned in 10 words or less.

Requirements:
- Focus on new inventions/technologies
- Maximum 10 words
- Clear, concise language
- Exclude company names

Format response as a JSON object with this schema:
{
  "patent": "brief description of patentable innovation"
}
""",
}


def get_narrative_system_prompt(theme_labels: list[str]) -> str:
    """Generate a system prompt for labeling sentences with narrative labels."""
    return narrative_system_prompt_template.format(
        theme_labels=theme_labels,
    )


def get_screener_system_prompt(
    main_theme: str, label_summaries: list[str], unknown_label: str
) -> str:
    """Generate a system prompt for labeling sentences with thematic labels."""
    return screener_system_prompt_template.format(
        main_theme=main_theme,
        label_summaries=label_summaries,
        unknown_label=unknown_label,
    )


risk_system_prompt_template: str = """

Forget all previous prompts.

You are assisting a professional analyst in evaluating both the exposure and risk classification for "Target Company" regarding the Risk Scenario "{main_theme}". This involves a two-step process: confirming exposure of "Target Company" and classifying specific risks if exposure is confirmed. Use the headline for contextual understanding.

<input_details>
You will receive the following information::
- ID: [text ID]
- Entity Sector: [The sector in which Target Company operates]
- Entity Industry: [The specific industry segment in which Target Company operates]
- Headline: [The Headline of the News Article containing Text]
- Text: [Paragraph requiring analysis]
- Risk Scenario: "{main_theme}"
</input_details>

Follow these guidelines:

<exposure_assessment>
- Examine whether the text explicitly mentions the Risk Scenario "{main_theme}" or any of its core components.
- Ensure that "Target Company" is the main focus of the text and that it is clearly stated that "Target Company" is facing or will face consequences caused by the Risk Scenario "{main_theme}".
- Assess if there are DIRECT consequences on "Target Company's" business activities, operations, or future performance.
- Designate the exposure as unclear if the text lacks an explicit DIRECT link between "Target Company" and the Risk Scenario
- Designate the exposure as unclear if the text relies on generic information.
</exposure_assessment>

<risk_classification>
If direct exposure of Target Company is confirmed:

- Identify and classify the specific risk using this list of Risk Sub-Scenarios:
    "{label_summaries}".

- Follow a detailed classification process:
    - Examine the text to confirm how the Risk Scenario "{main_theme}" directly impacts "Target Company" through one of the Risk Sub-Scenarios.
    - Write a concise motivation that explains the direct link between "Target Company" and the Risk Sub-Scenario as stated in the text.
    - The motivation should always start with "Target Company".
    - Consider the Entity Sector and Industry to align the Risk Sub-Scenario label with Target Company's operations, reflecting material risks faced according to the text.
    - Identify an appropriate Risk Sub-Scenario label from the list that describes explicitly the impact on the company's business, operations, or performance.
    - Be specific in the risk classification, ensure that the risk sub-scenario represents well your motivation statement.
    - Ensure that the Risk Sub-Scenario label can be directly extracted from the text that it describes with high granularity how "Target Company" is affected.
    - Avoid deriving conclusions based on unstated or inferred information. Focus only on the explicit content of the text or headline.
</risk_classification>

<verbatim_quotes_extraction>
- Extract verbatim quotes from the text that support the classification and illustrate Target Company's exposure to the specific Risk Sub-Scenario.
- Ensure quotes directly relate to the impact described and justify the risk label.
- Extract full sentences or phrases that clearly indicate, as standalone statements, how "Target Company" is affected by the Risk Scenario "{main_theme}" and the Sub-Scenario label assigned.
</verbatim_quotes_extraction>

<response_format>
Structure your response as a JSON object containing:
"sentence_id": "<sentence_id>"
"motivation": : A concise explanation describing the link between "Target Company" and the Risk Sub-Scenario.
"label": State the specific risk Sub-Scenario label or 'unclear'.
"quotes": Present verbatim quotes that justify exposure and risk label assignment.

{{"<sentence_id>": {{"motivation": "<motivation>", "label": "<risk_classification_label>", "quotes": "<verbatim_quotes>"}}}}.
</response_format>

<examples>
ID: 3
Entity Sector: Consumer Staples
Entity Industry: Food and Beverages
Headline: "Tariffs to Strain Supply Chains Globally"
Text: "New tariffs against China will significantly impact Target Company's operations due to its reliance on raw materials from Chinese suppliers."
Scenario: "New Tariffs against China"
Output:

{{3:{{
  "motivation": "Target Company's supply operations are directly impacted by new tariffs due to their reliance on raw materials sourced from China.",
  "label": "Supply Chain Disruption",
  "quotes": ["New tariffs against China will significantly impact Target Company's operations", "reliance on raw materials from Chinese suppliers"]}}
}}

ID: 5
Entity Sector: Financial Services
Entity Industry: Banking
Headline: "Interest Rate Fluctuations to Affect Markets"
Text: "Target Company's analysts are forecasting higher risks associated with potential interest rate changes."
Scenario: "Interest Rate Volatility"
Output:

{{5:{{
  "motivation": "Target Company is not directly affected by any risk associated with Interest Rate fluctuations.",
  "label": "unclear",
  "quotes": []
}}}}

ID: 2
Entity Sector: Retail
Entity Industry: Apparel
Headline: "Economic Challenges Ahead Due to Tariffs on China"
Text: "Target Company's analysts report a potential economic downturn linked to new tariffs against China."
Risk Scenario: "New Tariffs Against China"
Output:

{{2:{{
  "motivation": "Target Company is not said to be directly affected by new tariffs. Its analyst are simply working on a report assessing generic consequences",
  "label": "unclear",
  "quotes": []}}
}}

ID: 3
Entity Sector: Technology
Entity Industry: Software
Headline: "Analyzing External Factors in Business Strategy"
Text: "Target Company is studying external factors such as tariffs to gauge potential risks."
Risk Scenario: "New Tariffs on Semiconductors"
Output:

{{3:{{
  "motivation": "Target Company is merely studying the situation without asserting any direct impact on its operations.",
  "label": "unclear",
  "quotes": []}}
}}

ID: 4
Entity Sector: Finance
Entity Industry: Investment Banking
Headline: "Market Trends Influence Stock Performance"
Text: "Target Company's stock is influenced by broad market trends."
Risk Scenario: "Increased Uncertainty and Volatility"
Output:

{{4:{{
  "motivation": "The text does not related to the Risk Scenario and it does not mention any specific risk sub-scenario affecting Target Company.",
  "label": "unclear",
  "quotes": []}}
}}

ID: 5
Entity Sector: Manufacturing
Entity Industry: Automotive
Headline: "Tariffs and Their Economic Impact"
Text: "Target Company researchers estimate that tariffs will affect the broader economy."
Risk Scenario: "New Tariffs against China"
Output:

{{5:{{
  "motivation": "Target Company is not linked with any specific risk sub-scenario or any tangible effect of the Risk Scenario.",
  "label": "unclear",
  "quotes": []}}
}}

ID: 2
Entity Sector: Consumer Staples
Entity Industry: Food and Beverages
Headline: "China Tariffs Impact Supply Chains"
Text: "According to recent reports, Target Company is heavily dependent on China. The recent tariffs against China have forced Target Company to reconsider its supply chain, potentially leading to increased logistics costs."
Risk Scenario: "New Tariffs against China"
Output:

{{2:{{
  "motivation": "Target Company is said to be reconsidering its supply chain in the face of the risk scenario. The text clearly links Target Company with the Risk Scenario and mentions an explicit Sub-scenario risk of Supply Chain Disruptions.",
  "label": "Supply Chain Disruption",
  "quotes": [
    "Target Company is heavily dependent on China",
    "The recent tariffs against China have forced Target Company to reconsider its supply chain, potentially leading to increased logistics costs."
  ]}}
}}
</examples>

"""


def get_risk_system_prompt(main_theme: str, label_summaries: list[str]) -> str:
    """Generate a system prompt for labeling sentences with thematic labels."""
    return risk_system_prompt_template.format(
        main_theme=main_theme, label_summaries=label_summaries
    )


# Theme Validation Prompt Template
# Two-step analysis: 1) Verify theme relevance, 2) Assess Target Company impact
theme_validation_prompt_template: str = """
Forget all previous prompts.
You are assisting a professional analyst in evaluating whether a text is related to the theme '{main_theme}' and how it impacts "Target Company".

Your task involves two steps:

**Step 1 - Theme Relevance Check:**
- Each input consists of a sentence ID, labels (list), themes (list), and the sentence text.
- Analyze if the text explicitly discusses '{main_theme}' in connection with the provided labels and themes.
- The labels and themes provided for each text indicate the expected context - verify if the text actually matches this context.
- The text must clearly establish a connection between the main theme and the Target Company.
- Output: true if the text is genuinely related to the main theme according to the labels/themes, false otherwise.

**Step 2 - Impact Assessment (only if Step 1 is true):**
- Evaluate how the Target Company is affected in this context.
- This is similar to a sentiment analysis but focused on business impact.
- Possible values (use only these four):
  - "Positive": Target Company benefits from the situation described (e.g., competitive advantage, growth opportunity, positive market position)
  - "Negative": Target Company is negatively affected (e.g., increased costs, competitive threat, regulatory burden, market loss)
  - "Neutral": The impact is mentioned but neutral/factual without clear positive or negative direction
  - "Unclear": Cannot determine the impact direction from the text
- Provide a brief motivation explaining your assessment.

**Response Format:**
JSON object:
{{"<sentence_id>": {{"is_theme_related": <true/false>, "impact": "<impact>", "motivation": "<motivation>"}}, ...}}

**Important Notes:**
- If is_theme_related is false, set impact to "Unclear" and motivation should explain why the text is not related to the main theme.
- Labels and themes in the input may contain one or more items (they are lists).
- Focus only on explicit information in the text.
- The motivation should always start with "Target Company" when the text is related.
- Ensure all strings in the JSON are correctly formatted with proper quotes.
"""


def get_theme_validation_prompt(main_theme: str) -> str:
    """Generate a theme validation system prompt.

    Args:
        main_theme: The main theme to validate against.

    Returns:
        Formatted system prompt for theme validation.
    """
    return theme_validation_prompt_template.format(main_theme=main_theme)
