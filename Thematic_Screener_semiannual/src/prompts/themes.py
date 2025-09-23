def compose_themes_system_prompt(
    main_theme: str, analyst_focus: str = ""
) -> str:
    prompt = f"""
	Forget all previous prompts. 
	You are assisting a professional analyst tasked with creating a screener to measure the impact of the theme {main_theme} on companies. 
	Your objective is to generate a comprehensive tree structure of distinct sub-themes that will guide the analyst's research process.
	
	Follow these steps strictly:
	
	1. **Understand the Core Theme {main_theme}**:
	   - The theme {main_theme} is a central concept. All components are essential for a thorough understanding.
	
	2. **Create a Taxonomy of Sub-themes for {main_theme}**:
	   - Decompose the main theme {main_theme} into concise, focused, and self-contained sub-themes.
	   - Each sub-theme should represent a singular, concise, informative, and clear aspect of the main theme.
	   - Expand the sub-theme to be relevant for the {main_theme}: a single word is not informative enough.    
	   - Prioritize clarity and specificity in your sub-themes.
	   - Avoid repetition and strive for diverse angles of exploration.
	   - Provide a comprehensive list of potential sub-themes.
	  
	3. **Iterate Based on the Analyst's Focus {analyst_focus}**:
	   - If no specific {analyst_focus} is provided, transition directly to formatting the JSON response.
	
	4. **Format Your Response as a JSON Object**:
	   - Each node in the JSON object must include:
	     - `node`: an integer representing the unique identifier for the node.
	     - `label`: a string for the name of the sub-theme.
	     - `summary`: a string to explain briefly in maximum 15 words why the sub-theme is related to the theme {main_theme}.
	       - For the node referring to the first node {main_theme}, just define briefly in maximum 15 words the theme {main_theme}.
	     - `children`: an array of child nodes.
	
	## Example Structure:
	**Theme: Global Warming**
	
	{{
	    "node": 1,
	    "label": "Global Warming",
	    "children": [
	        {{
	            "node": 2,
	            "label": "Renewable Energy Adoption",
	            "summary": "Renewable energy reduces greenhouse gas emissions and thereby global warming and climate change effects",
	            "children": [
	                {{"node": 5, "label": "Solar Energy", "summary": "Solar energy reduces greenhouse gas emissions"}},
	                {{"node": 6, "label": "Wind Energy", "summary": "Wind energy reduces greenhouse gas emissions"}},
	                {{"node": 7, "label": "Hydropower", "summary": "Hydropower reduces greenhouse gas emissions"}}
	            ]
	        }},
	        {{
	            "node": 3,
	            "label": "Carbon Emission Reduction",
	            "summary": "Carbon emission reduction decreases greenhouse gases",
	            "children": [
	                {{"node": 8, "label": "Carbon Capture Technology", "summary": "Carbon capture technology reduces atmospheric CO2"}},
	                {{"node": 9, "label": "Emission Trading Systems", "summary": "Emission trading systems incentivize reductions in greenhouse gases"}}
	            ]
	        }},
	        {{
	            "node": 4,
	            "label": "Climate Resilience and Adaptation",
	            "summary": "Climate resilience adapts to global warming impacts, reducing vulnerability",
	            "children": [
	                {{"node": 10, "label": "Sustainable Agriculture", "summary": "Sustainable agriculture reduces emissions, enhancing food security amid climate change"}},
	                {{"node": 11, "label": "Infrastructure Upgrades", "summary": "Infrastructure upgrades enhance resilience and reduce emissions against climate change"}}
	            ]
	        }},
	        {{
	            "node": 12,
	            "label": "Biodiversity Conservation",
	            "summary": "Biodiversity conservation supports ecosystems",
	            "children": [
	                {{"node": 13, "label": "Protected Areas", "summary": "Protected areas preserve ecosystems, aiding climate resilience and mitigation"}},
	                {{"node": 14, "label": "Restoration Projects", "summary": "Restoration projects sequester carbon"}}
	            ]
	        }},
	        {{
	            "node": 15,
	            "label": "Climate Policy and Governance",
	            "summary": "Climate policy governs emissions, guiding efforts to combat global warming",
	            "children": [
	                {{"node": 16, "label": "International Agreements", "summary": "International agreements coordinate global efforts to reduce greenhouse gas emissions"}},
	                {{"node": 17, "label": "National Legislation", "summary": "National legislation enforces policies that reduce greenhouse gas emissions"}}
	            ]
	        }}
	    ]
	}}
    """
    return prompt.strip()
