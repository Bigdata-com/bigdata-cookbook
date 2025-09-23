def compose_risk_system_prompt_focus(main_theme: str, analyst_focus: str) -> str:
    prompt = f"""
            Forget all previous prompts.
            You are assisting a professional risk analyst tasked with creating a taxonomy to classify the impact of the Risk Scenario '**{main_theme}**' on companies.
            Your objective is to generate a **comprehensive tree structure** that maps the **risk spillovers** stemming from the Risk Scenario '**{main_theme}**', and generates related sub-scenarios. 

            Key Instructions:

            1. **Understand the Risk Scenario: '{main_theme}'**:
                - The Risk Scenario '**{main_theme}**' represents a central, multifaceted concept that may be harmful or beneficial to firms.
                - Your task is to identify how the Risk Scenario impacts firms through various **risk spillovers** and transmission channels.
                - Summarize the Risk Scenario '**{main_theme}**' in a **short list of essential keywords**.
                - The keyword list should be short (1-2 keywords). Avoid unnecessary, unmentioned, indirectly inferred, or redundant keywords.

            2. **Create a Tree Structure for Risk Spillovers and Sub-Scenarios**:
                - Decompose the Risk Scenario into **distinct, focused, and self-contained risk spillovers**.
                - Each risk spillover must represent a **specific risk channel** through which firms are exposed to as a consequence of the Risk Scenario.
                - Label each **primary node** in the tree explicitly as a "Risk" in the `Label` field. For example:
                    - Use "Cost Risk" instead of "Cost Impacts."
                    - Use "Supply Chain Risk" instead of "Supply Chain Disruptions."
                - Risk spillovers must:
                    - Cover a wide range of potential impacts on firms' operations, business, performance, strategy, profits, and long-term success.
                    - Explore both macroeconomic and microeconomic dimensions of the Risk Scenario '**{main_theme}**' and analyze their impact on firms when relevant.
                        - Microeconomic effects, such as cost of inputs, directly affect firms' operations
                        - Macroeconomic effects may affect firms revenues directly (e.g. currency fluctuations) or indirectly (e.g. economic downturns triggering lower demand).
                    - Include **direct and indirect consequences** of the main scenario.
                    - Represent **dimensions of risk** that firms must monitor or mitigate.
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
                    "Summary": "New tariffs could trigger retaliatory tariffs from China against US companies",
                    "Children": []
                }},
                {{
                    "Node": 4,
                    "Label": "US Dollar / Chinese Yuan Currency Volatility",
                    "Summary":"US firms face uncertainty in trade as new tariffs will increase volatility in the USD/CNY.",
                    "Children": []
                }}
                ]
            }}
            ],
            "Keywords": ["Tariffs", "China"]
        }}
        """ 
    return prompt.strip()
