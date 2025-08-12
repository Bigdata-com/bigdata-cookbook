import json
import re
import openai
import asyncio

class LexiconGenerator:
    def __init__(self, openai_key: str, model: str="gpt-4o", seeds: list[int]=None):
        self.openai_key = openai_key
        self.model = model
        if seeds is None:
            self.seeds = [123, 123456, 123456789, 456789, 789]
        else:
            self.seeds = seeds

    async def _fetch_keywords(self, theme, seed, rr0=None):
        client = openai.AsyncOpenAI(api_key=self.openai_key)
        system_prompt = f"""You are an expert tasked with generating a lexicon of the most important and relevant keywords specific to the {theme}.

        Your goal is to compile a list of terms that are critical for understanding and analyzing the {theme}. This lexicon should include only the most essential keywords, phrases, and abbreviations that are directly associated with {theme} topics, analysis, logistics, and industry reporting.

        Guidelines:

        1. **Focus on relevance:** Include only the most important and commonly used keywords that are uniquely tied to the {theme}. These should reflect key concepts, industry-specific mechanisms, benchmarks, logistical aspects, and terminology that are central to the theme.
        2. **Avoid redundancy:** Do not repeat the primary terms of the theme in multiple phrases. Include the main term (e.g., "{theme}") only as a standalone term, and focus on other specific terms without redundant repetition.
        3. **Strict exclusion of generic terms:** Exclude any terms that are generic or broadly used across different fields, such as "Arbitrage," "Hedge," "Liquidity," or "Futures Contract," even if they have a specific meaning within the context of {theme}. Only include terms that are uniquely relevant to {theme} and cannot be applied broadly.
        4. **Include specific variations:** Where applicable, provide both the full form and common abbreviations relevant to the {theme}. **Present the full term and its abbreviation as separate entries.** For example, instead of "Zero Lower Bound (ZLB)", list "Zero Lower Bound" and "ZLB" as separate keywords.
        5. **Ensure clarity:** Each keyword should be concise, clear, and directly relevant to the {theme}, avoiding any ambiguity.
        6. **Select only the most critical:** There is no need to reach a specific number of keywords. Focus solely on the most crucial terms without padding the list. If fewer keywords meet the criteria, that is acceptable.

        The output should be a lexicon of only the most critical and uniquely relevant keywords related to the {theme}, formatted as a JSON list, with full terms and abbreviations listed separately.
        """
        if rr0:
            print("Using example output for expansion:", rr0)
            system_prompt2 = f"{system_prompt}\nBelow is just an example output, please expand:\n{str(rr0)}"
        else:
            system_prompt2 = system_prompt
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt2},
                {"role": "user", "content": theme}
            ],
            temperature=0.25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=seed,
            response_format={"type": "json_object"}
        )
        return response.model_dump()['choices'][0]['message']['content']

    async def _generate(self, theme):
        keywords = {}
        rr0 = None
        #print("[LexiconGenerator] Using seeds:", self.seeds)
        tasks = [self._fetch_keywords(theme, seed, rr0) for seed in self.seeds]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for i, rr in enumerate(responses):
            if isinstance(rr, Exception):
                #print(f"[LexiconGenerator] Seed {self.seeds[i]} failed with error: {rr}")
                continue
            #print(f"[LexiconGenerator] Raw response for seed {self.seeds[i]}:", rr)
            rr = re.sub(r'```', '', rr)
            rr = re.sub(r'json', '', rr)
            try:
                concept_dict = json.loads(rr)
                #print(f"[LexiconGenerator] Parsed JSON for seed {self.seeds[i]}:", concept_dict)
                rr0 = concept_dict.copy()
                for s in concept_dict:
                    rr0[s] = [concept_dict[s][0]]
                for k in concept_dict:
                    keywords[k] = keywords.get(k, []) + concept_dict.get(k, [])
            except Exception as e:
                #print(f"[LexiconGenerator] Seed {self.seeds[i]} JSON error: {e}")
                continue
        #print(f"[LexiconGenerator] Combined keywords dict:", keywords)
        # Consolidate all keywords
        all_keywords = []
        for klist in keywords.values():
            all_keywords.extend(klist)
        #print(f"[LexiconGenerator] All keywords before deduplication:", all_keywords)
        # Remove duplicates, preserve order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        #print(f"[LexiconGenerator] Unique keywords after deduplication:", unique_keywords)
        #unique_keywords = [keyword.replace(' ', '-') for keyword in unique_keywords]
        return unique_keywords

    def generate(self, theme):
        """
        Synchronously generate a consolidated lexicon for a theme.
        """
        return asyncio.run(self._generate(theme))