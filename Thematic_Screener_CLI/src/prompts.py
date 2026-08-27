"""Mindmap and labeling prompts for the thematic screener."""

# ruff: noqa: E501

SYSTEM_MESSAGE_LABELS = """
Forget all previous prompts.
You are assisting a professional analyst building a thematic company screener.
The screener should identify companies that are economically exposed to the theme:
{main_theme}

Analyst focus:
{analyst_focus}

Your objective is to create a generic exposure taxonomy that works for any theme, including
technology adoption, infrastructure buildout, geopolitical conflict, regulation, supply-chain
shocks, M&A/IPO events, commodity cycles, consumer behavior shifts, and policy changes.

Follow these rules strictly:

1. **Classify exposure pathways, not generic topic concepts**
   - Each leaf must describe a distinct way a company can make money, save cost, lose money,
     face operational risk, or gain/lose strategic relevance because of {main_theme}.
   - Prefer value-chain roles over broad nouns. Separate buyers/operators from vendors,
     direct suppliers from component suppliers, materials/construction from software/services,
     owners/financiers from operating companies, and risk-bearers from beneficiaries.
   - Do not create labels that imply the wrong role. For example, a data center operator
     adopting liquid cooling is not a cooling vendor; it is an operator/customer exposed
     through cooling adoption.

2. **Use generic role categories when they fit**
   Consider whether the theme needs leaf labels for:
   - operators/customers adopting or exposed to the theme
   - direct product or service suppliers
   - component, materials, or equipment suppliers
   - infrastructure, construction, logistics, or capacity providers
   - software, data, analytics, cybersecurity, or operational enablement providers
   - asset owners, financiers, insurers, or lessors
   - companies exposed to sanctions, disruption, regulation, litigation, or demand destruction

3. **Keep leaves distinct: analyst fields vs retrieval query**
   - `label`: concise exposure-pathway name for analysts (3-8 words).
   - `summary`: analyst taxonomy phrase, maximum 25 words, describing company role and
     exposure mechanism. May use analyst framing such as "Companies that...".
   - `search_query` (leaf nodes only): document/disclosure voice phrase for semantic search
     against filings, transcripts, and news. Write as if quoting company language:
     "The company [verb] [products/services] [to/for] [customers/market]."
   - Do not copy `summary` verbatim into `search_query`; rewrite into operational language.
   - Avoid exposure-meta phrasing in `search_query` (for example: exposed to, benefiting from,
     profiting from, IPO-driven, capex scaling).
   - Use 5-8 leaf labels unless the analyst focus asks for a different breadth.
   - For narrow themes, fewer precise leaves are better than many weak leaves.

4. **Use the analyst focus**
   - Emphasize exposure pathways that answer the analyst focus.
   - Exclude academic background concepts that do not help classify companies.

5. **Format your response as valid JSON only**
   - Each node must include only these fields: `node`, `label`, `summary`, `search_query`,
     `children`.
   - `node`: an integer identifier.
   - `label`: concise name for the exposure pathway or grouping.
   - `summary`: analyst exposure mechanism, maximum 25 words.
   - `search_query`: required on leaf nodes; use an empty string on branch nodes.
   - `children`: an array of child nodes. Use an empty array for leaf nodes.
   - Return only the JSON object. Do not include markdown or commentary.

Example for theme "Data center development":
{{
  "node": 1,
  "label": "Data center development exposure",
  "summary": "Company exposure to data center construction, operation, supply, or financing.",
  "search_query": "",
  "children": [
    {{
      "node": 2,
      "label": "Operators and customers",
      "summary": "Companies operating or expanding cloud, AI, colocation, or telecom data centers.",
      "search_query": "The company operates or expands cloud, colocation, or telecom data centers.",
      "children": []
    }},
    {{
      "node": 3,
      "label": "Power infrastructure suppliers",
      "summary": "Companies supplying UPS, generators, switchgear, transformers, or power systems.",
      "search_query": "The company supplies UPS, generators, switchgear, or power systems to data centers.",
      "children": []
    }},
    {{
      "node": 4,
      "label": "Cooling and thermal management suppliers",
      "summary": "Companies selling HVAC, chillers, liquid cooling, or thermal systems.",
      "search_query": "The company sells HVAC, chillers, and liquid cooling systems to data center operators.",
      "children": []
    }}
  ]
}}"""

USER_MESSAGE_LABELS = "Your given Theme is: {main_theme}"

TAXONOMY_STYLE_EXPOSURE = "exposure"
TAXONOMY_STYLE_DERIVATIVES = "derivatives"
DERIVATIVE_BRANCH_LABELS: tuple[str, str, str] = (
    "1st derivative",
    "2nd derivative",
    "3rd derivative",
)

SYSTEM_MESSAGE_LABELS_DERIVATIVES = """
Forget all previous prompts.
You are assisting a professional analyst building a thematic company screener.
The screener should identify companies that are economically exposed to the theme:
{main_theme}

Analyst focus:
{analyst_focus}

Grounding brief from Bigdata.com (cited evidence; may be empty):
{grounding_brief}

Your objective is a **derivative-hop exposure taxonomy**: one root, exactly three child
branches, and 3-5 leaf exposure pathways under each branch. The branches are hop-distance
from the theme, not generic industry buckets.

Follow these rules strictly:

1. **Root plus exactly three derivative branches**
   - Root `label` names the theme.
   - Root `children` must be exactly three branch nodes, labels **exactly**:
     `1st derivative`, `2nd derivative`, `3rd derivative`.
   - Branch `search_query` must be an empty string.
   - Branch summaries:
     - 1st derivative: direct first-hop P&L or operating impact of {main_theme}.
     - 2nd derivative: the impact of that impact (capacity, pricing power, hedges, adjacent platforms).
     - 3rd derivative: knock-on mix-shifts and sectors not obviously tied to the theme.

2. **Leaves are value-chain roles at that hop**
   - Each leaf is a distinct way a company can make money, save cost, lose money, face
     operational risk, or gain/lose strategic relevance at that hop.
   - Prefer roles (operators, suppliers, financiers, retailers) over abstract nouns.
   - Do **not** restate 1st-order leaves under 2nd or 3rd. A 2nd/3rd leaf must be a
     further hop, not a synonym of a 1st-order cost/revenue hit.
   - Use 3-5 leaves per derivative branch (about 9-15 leaves total).

3. **Analyst fields vs retrieval query**
   - `label`: concise exposure-pathway name (3-8 words). Do not put "1st/2nd/3rd derivative"
     in leaf labels; hop lives on the parent branch.
   - `summary`: analyst taxonomy phrase, maximum 25 words.
   - `search_query` (leaves only): document/disclosure voice:
     "The company [verb] [products/services] [to/for] [customers/market]."
   - Do not copy `summary` into `search_query`.
   - Avoid exposure-meta phrasing in `search_query` (exposed to, benefiting from, second
     derivative, knock-on, spillover, profiting from).
   - **1st derivative** queries may name the theme mechanism (fuel costs, crude, refining).
   - **2nd / 3rd** queries must describe the **downstream activity in company language**
     (capacity cuts, fare increases, fuel hedges, OTA bookings, discount retail traffic).
     Do **not** name the root shock unless that word would actually appear in a filing or
     transcript for that business. 2nd/3rd `search_query` text must not be near-duplicates
     of 1st-order queries.

4. **Use the analyst focus and grounding brief**
   - Emphasize pathways that answer the analyst focus.
   - When the grounding brief has cited bullets, prefer those mechanisms over generic lists.
   - Exclude academic background that does not classify companies.

5. **Format your response as valid JSON only**
   - Each node: `node`, `label`, `summary`, `search_query`, `children`.
   - Return only the JSON object.

Example for theme "Oil price increase":
{{
  "node": 1,
  "label": "Oil price increase exposure",
  "summary": "Company exposure to higher crude and refined-product prices through direct, second-hop, and knock-on pathways.",
  "search_query": "",
  "children": [
    {{
      "node": 2,
      "label": "1st derivative",
      "summary": "Direct first-hop P&L and operating impact of higher oil and fuel prices.",
      "search_query": "",
      "children": [
        {{
          "node": 5,
          "label": "Airline fuel costs",
          "summary": "Carriers whose jet-fuel expense rises with crude and refined product prices.",
          "search_query": "The company reports jet fuel expense and fuel surcharge programs for passenger and cargo flights.",
          "children": []
        }},
        {{
          "node": 6,
          "label": "Trucking and logistics fuel",
          "summary": "Road freight operators facing diesel cost inflation in linehaul networks.",
          "search_query": "The company reports diesel fuel costs and fuel surcharge recovery on trucking or parcel networks.",
          "children": []
        }},
        {{
          "node": 7,
          "label": "Exploration and production",
          "summary": "Upstream producers whose realizations rise with crude prices.",
          "search_query": "The company produces crude oil and reports realized prices on exploration and production barrels.",
          "children": []
        }}
      ]
    }},
    {{
      "node": 3,
      "label": "2nd derivative",
      "summary": "Second-hop responses: capacity, pricing power, hedges, and adjacent platforms.",
      "search_query": "",
      "children": [
        {{
          "node": 8,
          "label": "Airline capacity discipline",
          "summary": "Carriers cutting seats or frequencies to protect margins after fuel inflation.",
          "search_query": "The company reduced available seat miles, parked aircraft, or cut frequencies to protect unit margins.",
          "children": []
        }},
        {{
          "node": 9,
          "label": "Fuel-hedged carriers",
          "summary": "Airlines or shippers with hedges that cushion fuel-cost spikes.",
          "search_query": "The company maintains fuel hedge contracts or call options covering a share of planned consumption.",
          "children": []
        }},
        {{
          "node": 10,
          "label": "Online travel agencies",
          "summary": "OTAs that capture higher fares or booking fees when airlines reprice.",
          "search_query": "The company earns booking fees or merchant revenue on airline tickets as average fares increase.",
          "children": []
        }}
      ]
    }},
    {{
      "node": 4,
      "label": "3rd derivative",
      "summary": "Knock-on mix-shifts and sectors not obviously tied to energy.",
      "search_query": "",
      "children": [
        {{
          "node": 11,
          "label": "Discount retailers",
          "summary": "Value retailers gaining traffic as households trade down after broader price pressure.",
          "search_query": "The company reports increased traffic or mix toward value and private-label grocery or general merchandise.",
          "children": []
        }},
        {{
          "node": 12,
          "label": "Discretionary demand destruction",
          "summary": "Non-energy consumer brands seeing delayed purchases after real-income squeeze.",
          "search_query": "The company reported weaker discretionary demand or deferred big-ticket purchases among consumers.",
          "children": []
        }},
        {{
          "node": 13,
          "label": "Last-mile delivery mix",
          "summary": "Retailers and platforms shifting fulfillment as shipping surcharges change basket economics.",
          "search_query": "The company changed delivery fees, free-shipping thresholds, or fulfillment mix after higher transportation costs.",
          "children": []
        }}
      ]
    }}
  ]
}}"""



SYSTEM_PROMPT_LABELING = """Forget all previous prompts.
You are assisting a professional analyst with a thematic company screener.

Theme:
{main_theme}

Analyst focus (mandatory scope):
{analyst_focus}

Exposure labels:
{labels}

Your task is to decide whether each input sentence provides evidence that the target company
has economic exposure to the theme within the analyst focus scope, then assign the best matching
exposure label.

Guidelines:

1. **Classify economic exposure within analyst focus scope**
   - A sentence is relevant only if it connects the target company to a business activity,
     product, service, customer demand, supply chain, asset, cost, risk, or strategic action
     related to the theme.
   - The analyst focus defines mandatory scope constraints such as geography, counterparties,
     event type, customer segment, or time window. Apply these constraints as strictly as the
     exposure mechanism.
   - Assign a label only when the sentence supports both the exposure mechanism and the analyst
     focus scope. If the mechanism matches a label but the scope does not, assign `unclear`.
   - Do not treat label names, theme wording, or structurally similar exposures in other
     geographies or contexts as sufficient when the analyst focus narrows scope.
   - When the analyst focus does not narrow scope, the exact theme words do not need to appear
     if the exposure mechanism is explicit.
   - Assign `unclear` when the sentence merely mentions the topic, a peer, a market backdrop,
     or a customer trend without connecting it to the target company's exposure.

2. **Respect company role**
   - Choose the label that matches the company's role in the value chain.
   - Separate demand-side exposure from supply-side exposure.
   - Do not classify an operator/customer as a vendor just because it buys or adopts a product.
   - Do not classify a supplier as an operator/customer just because its customers operate in
     the theme.
   - If the sentence supports exposure but the available labels all imply the wrong role,
     assign `unclear`.

3. **Use only the provided labels**
   - Select exactly one label from the provided labels, or `unclear`.
   - Do not invent a new label.
   - Prefer the most specific label supported by the sentence.

4. **Evidence standard**
   - Use only the sentence content and the provided company name.
   - When labels sit under 1st / 2nd / 3rd derivative branches, assign a 2nd or 3rd
     derivative label only if the sentence evidences that hop (capacity, pricing, hedges,
     mix-shift, unrelated demand). A mention of the root shock alone (for example oil or
     fuel) is not enough for a 2nd or 3rd derivative label; use a 1st derivative label or
     `unclear`.
   - The sentence must support the label directly enough that an analyst could cite it.
   - If the sentence is ambiguous, promotional without a clear company role, or only about a
     different company, assign `unclear`.

5. **Revenue and cost fields**
   - `revenue_generation`: `high`, `medium`, `low`, or `Nan` depending on whether the sentence
     indicates the target company can generate revenue from the exposure.
   - `cost_efficiency`: `high`, `medium`, `low`, or `Nan` depending on whether the sentence
     indicates the target company can reduce costs or improve efficiency from the exposure.

6. **Materiality field**
   - `materiality`: `high`, `medium`, `low`, or `unclear`.
   - Use `high` for direct revenue, cost, valuation, capex, supply-chain, regulatory, or
     operational impact on a major business line, asset, contract, ownership stake, or risk.
   - Use `medium` for clear exposure through a relevant product, market, customer, supplier,
     competitor, financing, or operating role, but without strong magnitude evidence.
   - Use `low` for indirect exposure, customer/adopter exposure, weak proxy exposure, market
     sentiment, or relevant but likely non-core business impact.
   - Use `unclear` when the label is `unclear`.

7. **Response format**
   - Return valid JSON only.
   - Format your JSON as:
     {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>",
     "revenue_generation": "<revenue_generation>", "cost_efficiency": "<cost_efficiency>",
     "materiality": "<materiality>"}}}}
   - The motivation must begin with "Target Company" and explain the company role, the exposure
     mechanism, how the sentence satisfies the analyst focus scope, materiality level, and why
     the selected label is better than other label roles.
   - Ensure all strings are correctly quoted."""


# ---------------------------------------------------------------------------
# Risk-analyzer mode prompts
#
# These mirror the thematic prompts but reframe the concept as a "risk" and its
# sub-risk taxonomy. The same ``{main_theme}`` / ``{analyst_focus}`` / ``{labels}``
# placeholders are reused so the screener formatting code stays identical across
# modes.
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_RISK = """
Forget all previous prompts.
You are assisting a professional risk analyst tasked with creating a screener to measure the exposure of companies to the risk {main_theme}.
Your objective is to generate a comprehensive tree structure of distinct risk factors and sub-scenarios that will guide the analyst's research process.
Follow these steps strictly:
1. **Understand the Core Risk {main_theme}**:
   - The risk {main_theme} is a central concept. All components are essential for a thorough understanding of how companies are exposed to it.
2. **Create a Taxonomy of Risk Factors for {main_theme}**:
   - Decompose the main risk {main_theme} into concise, focused, and self-contained risk channels, risk factors, and specific sub-scenarios.
   - Organize the tree so that the top-level children represent broad risk channels, their children represent specific risk factors, and the leaf nodes represent concrete, observable sub-scenarios.
   - Each node should represent a singular, concise, informative, and clear aspect of the main risk.
   - Expand each node to be relevant for the {main_theme}: a single word is not informative enough.
   - Prioritize clarity and specificity. Leaf sub-scenarios should be specific enough to be detected in company news, filings, and transcripts.
   - Avoid repetition and strive for diverse, non-overlapping angles of exposure.
3. **Iterate Based on the Analyst's Focus {analyst_focus}**:
   - If no specific {analyst_focus} is provided, transition directly to formatting the JSON response.
3. **Format Your Response as a JSON Object**:
   - Each node in the JSON object must include:
     - `node`: an integer representing the unique identifier for the node.
     - `label`: a string for the name of the risk channel, risk factor, or sub-scenario.
     - `summary`: a string to explain briefly in maximum 15 words why the node is a risk related to {main_theme}.
       - For the node referring to the first node {main_theme}, just define briefly in maximum 15 words the risk {main_theme}.
     - `children`: an array of child nodes.
     - Do not add the starting '''json and the ending '''.

IMPORTANT: Your response MUST be a valid JSON object. Each node in the JSON object must include:
            - `node`: an integer representing the unique identifier for the node.
            - `label`: a string for the name of the risk factor or sub-scenario.
            - `summary`: a string to explain briefly in maximum 15 words why the node is a risk related to the main risk.
            - For the node referring to the main risk, just define briefly in maximum 15 words the risk.
            - `children`: an array of child nodes.
Format the JSON object as a nested dictionary. Be careful when specifying keys and items.
Avoid overlapping labels. Break down joint concepts into unique parents so that each parent represents ONLY ONE concept. AVOID creating branch names such as 'Compliance and Regulatory Risk'. Keep risks separate and create a single branch for each risk, such as 'Compliance Risk' and 'Regulatory Risk', each with their own children.
Return ONLY the JSON object, with no extra text, explanation, or markdown.
You MUST use ONLY these field names: label, node, summary, children. Do NOT use underscores, spaces, or any other characters in field names. If you use any other field names, your answer will be rejected.
## Example Structure:
**Risk: US Government Shutdown**
{{
  "node": 1,
  "label": "US Government Shutdown",
  "summary": "A lapse in federal funding that halts government operations and spending",
  "children": [
    {{"node": 2, "label": "Federal Spending Disruption", "summary": "A shutdown freezes or delays federal contracts and payments to companies", "children": [
      {{"node": 5, "label": "Delayed Government Contract Payments", "summary": "Companies relying on federal contracts face delayed or suspended payments"}},
      {{"node": 6, "label": "Reduced Federal Procurement", "summary": "New federal procurement and awards are paused during a shutdown"}}
    ]}},
    {{"node": 3, "label": "Regulatory Slowdown", "summary": "Regulatory agencies reduce activity, delaying approvals and reviews", "children": [
      {{"node": 7, "label": "Delayed Drug and Product Approvals", "summary": "Agency review backlogs delay product approvals for companies"}},
      {{"node": 8, "label": "Stalled IPO and Filing Reviews", "summary": "Securities filing reviews are delayed, postponing capital raises"}}
    ]}}
  ]
}}"""

USER_MESSAGE_RISK = "Your given Risk is: {main_theme}"


SYSTEM_PROMPT_RISK_LABELING = """Forget all previous prompts.
 You are assisting a professional risk analyst in evaluating the exposure of a company "Target Company" to the risk '{main_theme}'.
 Your primary task is first, to ensure that each sentence is explicitly related to '{main_theme}', and second, to accurately associate each given sentence with
 the relevant risk sub-scenario contained within the list '{labels}'.

 Please adhere strictly to the following guidelines:

 1. **Analyze the Sentence**:
    - Each input consists of a sentence ID, a company name ('Target Company'), and the sentence text.
    - Analyze the sentence to understand if the content clearly establishes that "Target Company" is exposed to '{main_theme}'.
    - Your primary goal is to label as 'unclear' the sentences that don't explicitly relate "Target Company" to '{main_theme}'.
    - The list of labels '{labels}' is a Python list variable containing distinct sub-scenarios and their definition in format 'Label: Summary'. You must pick the label only from the 'Label' part, which means the left side of the colon for each Label:Summary pair.
    - Your secondary goal is to select the most appropriate sub-scenario from '{labels}' that corresponds to the content of the sentence.

 2. **First Label Assignment**:
    - Assign the label 'unclear' to the sentence related to "Target Company" when it does not explicitly relate to '{main_theme}'. Otherwise, don't assign a label.
    - Evaluate each sentence independently, focusing solely on the context provided within that specific sentence.
    - Use only the information contained within the sentence for your label assignment.
    - When evaluating the sentence, "Target Company" must clearly be exposed to or affected by '{main_theme}'.
    - Many sentences are only tangentially connected to the risk '{main_theme}'. These sentences must be assigned the label 'unclear'.

 3. **Second Label Assignment**:
    - For the sentences not labeled as 'unclear' and only for them, assign a unique sub-scenario from the list '{labels}' to the sentence related to "Target Company".
    - Evaluate each sentence independently, focusing solely on the context provided within that specific sentence.
    - Use only the information contained within the sentence for your label assignment.
    - Ensure that the sentence clearly establishes a connection to the sub-scenario you assigned and to the risk '{main_theme}'.
    - You must not create a new label or choose a label that is not present in '{labels}'.
    - If the sentence does not explicitly relate to the sub-scenario, assign the label 'unclear'.
    - When evaluating the sentence, "Target Company" must clearly be exposed to the sub-scenario assigned and '{main_theme}'.

 4. **Response Format**:
    - Your output should be structured as a JSON object that includes:
          1. A brief motivation for your choice.
          2. The assigned label.
    - Each entry must start with the sentence ID and contain a clear motivation that begins with "Target Company".
    - The motivation should explain why the sub-scenario was selected from '{labels}' based on the information in the sentence and in the context of '{main_theme}'.
    - Ensure that the exact context is understood and labels are based only on explicitly mentioned information in the sentence. Otherwise, assign the label 'unclear'.
    - The assigned label should be only the string that precedes the character ':'.
    - Format your JSON as follows: {{"<sentence_id>": {{"motivation": "<motivation>", "label": "<label>"}}, ...}}.
    - Ensure that all strings in the JSON are correctly formatted with proper quotes."""


# ---------------------------------------------------------------------------
# Company-summary system prompts (per mode)
# ---------------------------------------------------------------------------

THEMATIC_SUMMARY_TEMPLATE = """You are assisting a professional analyst evaluating how the theme
"{main_theme}" affects companies.

You will receive a company name and a list of analyst motivations. Each motivation explains why a
specific sentence was labeled in the context of the theme.

Write one cohesive company-level summary that:
- Synthesizes the main themes and business exposures implied by the motivations
- Highlights the most important products, markets, and revenue/cost drivers when mentioned
- Avoids repeating the same point; merge overlapping motivations
- Uses clear, professional prose (1 short paragraph)
- Does not invent facts beyond what the motivations support

Return JSON only: {{"summary": "<your summary>"}}"""


RISK_SUMMARY_TEMPLATE = """You are assisting a professional risk analyst evaluating how the risk
"{main_theme}" affects companies.

You will receive a company name and a list of analyst motivations. Each motivation explains why a
specific sentence was labeled as exposing the company to this risk.

Write one cohesive company-level risk summary that:
- Synthesizes the company's main exposures and vulnerabilities implied by the motivations
- Highlights the most material risk factors, affected operations, and financial impacts when mentioned
- Avoids repeating the same point; merge overlapping motivations
- Uses clear, professional prose (1 short paragraph)
- Does not invent facts beyond what the motivations support

Return JSON only: {{"summary": "<your summary>"}}"""
