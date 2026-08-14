"""Report generator using OpenAI (no SDK)."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import pandas as pd
from openai import AsyncOpenAI


class SummaryGenerator:
    """
    A class to generate summaries and reports from credit rating data.
    
    This class encapsulates the functionality for processing credit rating data,
    generating summaries, and creating structured reports through LLM processing.
    Uses the bigdata_research_tools.llm framework for LLM interactions with
    specialized handling for token limits through text splitting.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        system_prompt: str | None = None,
        max_workers: int = 30,
        api_key: str | None = None,
    ):
        """Initialize with OpenAI client.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini")
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
            max_workers: Maximum concurrent workers
            api_key: OpenAI API key (defaults to OPENAI_API_KEY)
        """
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.temperature = temperature
        self.max_workers = max_workers
        
        # Set default prompts
        self.system_prompt = system_prompt or """
**Task: Generate a Narrative Risk Summary for an Entity**

You are tasked with producing a clear, concise, and insightful narrative summary for a single entity based on a sequence of news extracts, risk channels, risk factors, and quotes. Your output should explain why the entity is receiving negative coverage related to refinancing risk in real estate. Identify and contextualize key risks, assess severity, causes, and consequences, and extract actionable insights.

**Input Structure:**

- `Entity`: [Entity]
- `Date`: [YYYY-MM-DD hh:mm:ss]
- `Headline`: [News headline]
- `Risk Channel`: [Risk channel]
- `Risk Factor`: [Risk factor]
- `Quote`: [Relevant news text or excerpt]

**Instructions:**

1. **Identify and Contextualize Risks**:
   - Focus on the main risks and negative coverage discussed in the news as related to real estate and refinancing risk.
   - Explain the context (market, regulatory, macroeconomic) and why these risks are relevant now.
   - If similar risks are repeated, consolidate them for clarity and avoid redundancy.

2. **Assess Severity, Causes, and Consequences**:
   - For each risk, explain its severity, underlying causes, and potential consequences for the entity and its stakeholders.
   - Compare the entity’s situation to peers or global trends if mentioned in the text.

3. **Extract Insights and Trends**:
   - Highlight actionable insights, signals, or trends that emerge from the news coverage.
   - Note any changes in risk level, sentiment, or outlook over time.

4. **Content Structure**:
   - Return a single, well-structured paragraph as a string.
   - Use clear, analytical language suitable for a professional risk report.
   - Reference specific dates, headlines, and quotes where they add clarity.

5. **Output Format**:
   - Return only the narrative summary string for the entity.
   - Do not include a timeline, bullet points, or any formatting other than a single paragraph.

**Example Output**:
Dubai’s real estate market faces elevated bubble risk, as highlighted by multiple sources in September 2025. News coverage points to surging property prices, population growth, and limited supply as key drivers, with UBS and Fitch warning of a possible correction by 2026. Oversupply, affordability pressures, and reliance on foreign investment further intensify vulnerabilities. Local court decisions on mortgage enforcement and rising construction activity add complexity, while sentiment-driven market dynamics could trigger abrupt shifts. Compared to other global cities, Dubai’s risk increase is among the strongest, raising concerns about the sustainability of recent gains and the potential for sudden downturns.

Use this structure for each entity, ensuring clarity, context, and actionable insight in your summary. Return only the summary string.
""".strip()

        self.consolidation_prompt = """
**Task: Consolidate Multiple Narrative Risk Summaries for an Entity**

You are an expert financial analyst. Your task is to merge several narrative summary completions about a single entity into one clear, concise, and insightful summary paragraph.

**Instructions:**

1. **Consistency and Clarity**:
   - Ensure the final summary is coherent, logically structured, and free from contradictions.
   - Seamlessly integrate information, especially where overlaps or repeated risks occur.

2. **Preserve Unique Insights**:
   - Include all unique and relevant details from each completion, ensuring nothing important is lost.
   - Eliminate redundant or repeated information. If similar risks or insights are described in multiple completions, merge them for clarity.

3. **Contextualization and Depth**:
   - Maintain analytical depth: explain why the entity is receiving negative coverage, identify and contextualize key risks, assess severity, causes, and consequences, and extract actionable insights.
   - Reference specific dates, headlines, and quotes only where they add clarity.

4. **Content Structure**:
   - Return a single, well-structured paragraph as a string.
   - Use clear, analytical language suitable for a professional risk report.

5. **Output Format**:
   - Return only the consolidated narrative summary string for the entity.
   - Do not include a timeline, bullet points, or any formatting other than a single paragraph.

**Example Output**:
Dubai’s real estate market faces elevated bubble risk, as highlighted by multiple sources in September 2025. News coverage points to surging property prices, population growth, and limited supply as key drivers, with UBS and Fitch warning of a possible correction by 2026. Oversupply, affordability pressures, and reliance on foreign investment further intensify vulnerabilities. Local court decisions on mortgage enforcement and rising construction activity add complexity, while sentiment-driven market dynamics could trigger abrupt shifts. Compared to other global cities, Dubai’s risk increase is among the strongest, raising concerns about the sustainability of recent gains and the potential for sudden downturns.


Carefully merge all completions into a single, clear, and actionable summary. Return only the summary string.
"""

    def _split_text_on_nearest_linebreak(self, text_string: str, num_splits: int) -> list[str]:
        """
        Split text into parts at the nearest line breaks for handling large inputs.
        
        Args:
            text_string: Text to split
            num_splits: Number of splits to create
            
        Returns:
            List of text segments
        """
        split_texts = [text_string]
        
        for _ in range(num_splits - 1):
            new_splits = []
            for text in split_texts:
                mid_index = len(text) // 2
                
                # Find nearby line breaks
                before_split = text.rfind('\n', 0, mid_index)
                after_split = text.find('\n', mid_index)
                
                # Choose split point
                if before_split != -1:
                    split_index = before_split
                elif after_split != -1:
                    split_index = after_split
                else:
                    split_index = mid_index
                
                # Split the text
                string1 = text[:split_index]
                string2 = text[split_index:]
                
                # Add context from first part to second part
                start_of_string1 = string1.split('\n')[:3]
                last_part_of_string1 = string1.split('\n')[-3:]
                
                context = '\n'.join(start_of_string1) + '\n'.join(last_part_of_string1) + '\n'
                string2 = context + string2
                
                new_splits.extend([string1, string2])
            
            split_texts = new_splits
        
        return split_texts

    def prepare_narrative_summary_input(self, df: pd.DataFrame,
                                        entity_name: str,
                                    date_col: str = 'date',
                                    sentence_id_col: str = 'sentence_id',
                                    text_col: str = 'text',
                                    summary_input: list = None) -> pd.DataFrame:
        """
        Generate summaries grouped by date from a DataFrame.
        
        Args:
            df: Input DataFrame
            date_col: Column name for dates
            sentence_id_col: Column name for sentence IDs
            text_col: Column name for text to summarize
            summary_input: Fields to include in summaries
            
        Returns:
            DataFrame with date-grouped summaries
        """
        # Handle empty values
        df = df.fillna('None').map(lambda x: "; ".join(x) if isinstance(x, list) else x)
        
        # Default fields: all columns except sentence ID
        if summary_input is None:
            fields_for_summary = [text_col]
        else:
            fields_for_summary = summary_input

        if text_col not in fields_for_summary:
            fields_for_summary.append(text_col)

        # Handle empty DataFrame
        if len(df) == 0:
            return f'Entity: {entity_name}\nNo data available for this period.'

        if len(df)>1:

            # Group by date and sentence ID - Aggregate at the chunk level
            def aggregate_sentence_fields(group):
                """Consolidate fields for the same date and sentence ID."""
                aggregated = {col: "; ".join(filter(None, group[col].unique())) 
                            for col in group.columns if col in fields_for_summary}
                return pd.Series(aggregated)

            sentence_grouped = df.groupby([date_col, sentence_id_col], dropna=False).apply(aggregate_sentence_fields, include_groups=False).reset_index()

            # Create sentence input summaries
            def create_sentence_summary(row):
                """Create structured summary for a sentence."""
                summary = [f"{field.replace('_', ' ').title()}: {row[field]}" 
                        for field in fields_for_summary if row[field]]
                return "\n".join(summary)
                
            sentence_grouped['sentence_summary'] = sentence_grouped.apply(create_sentence_summary, axis=1)
            
            # Group by date to consolidate sentences - Aggregate all chunks from the same day
            def aggregate_date_fields(group):
                """Consolidate sentences for the same date."""
                aggregated = {col: "; ".join(filter(None, group[col].unique())) 
                            for col in group.columns if col not in [date_col, 'sentence_summary']}
                aggregated['summary_input'] = "\n".join(group['sentence_summary'])
                return pd.Series(aggregated)
                
            date_grouped = sentence_grouped.groupby(date_col, dropna=False).apply(aggregate_date_fields,include_groups=False).reset_index()
            # date_grouped['id'] = range(len(date_grouped))
            # date_grouped['summary_input'] = date_grouped.apply(lambda row: 'Id: ' + str(row.id) + '\n' + row['summary_input'], axis=1) not needed if I create the prompts as in the labeler

            report_text_input = f'Entity: {entity_name}\n' + '\n'.join(
                [f'Date: {str(row[date_col])}\n{row.summary_input}' 
                for i, row in date_grouped.iterrows()]
            )
            return report_text_input
        else:
            return f'Entity: {entity_name}\n' + f'Date: {df[date_col].iloc[0]}\n' + '\n'.join([df[field].iloc[0] for field in fields_for_summary])

    async def _get_response(self, messages: list[dict[str, str]]) -> str:
        """Call OpenAI and return response text."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def summarize_string(
        self,
        text: str,
        system_prompt: str | None = None,
        max_retries: int = 5,
        max_split_retries: int = 5,
    ) -> str:
        """
        Summarize a text string with retry and text splitting capabilities.
        
        Args:
            text: Text to summarize
            prompt_type: Type of prompt to use
            replacements: Key-value pairs for prompt replacements
            max_retries: Maximum retry attempts
            max_split_retries: Maximum retry attempts with text splitting
            
        Returns:
            Summarized text
        """
        if system_prompt is None:
            system_prompt = self.system_prompt
        # print(system_prompt)
        # print(text)

        # Build chat history
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Try to get response directly
        try:
            return asyncio.run(self._get_response(chat_history))
        except Exception as e:
            if 'context_length_exceeded' in str(e) or 'string_above_max_length' in str(e):
                print("Text too long for direct processing, attempting split-and-consolidate approach...")
                
                # Try splitting and processing in chunks
                try:
                    # Split the text
                    splits = self._split_text_on_nearest_linebreak(text, 2)  # Start with 2 splits
                    
                    # Process each split
                    results = []
                    for split_text in splits:
                        split_chat_history = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": split_text}
                        ]
                        response = asyncio.run(self._get_response(split_chat_history))
                        results.append(response)
                    
                    # Consolidate results
                    if len(results) > 1:
                        consolidation_prompt = self.consolidation_prompt
                        consolidation_text = '\n\n'.join([f"Completion {i+1}: {comp}" for i, comp in enumerate(results)])
                        
                        consolidation_chat_history = [
                            {"role": "system", "content": consolidation_prompt},
                            {"role": "user", "content": f"Please merge and consolidate these completions into a single response:\n\n{consolidation_text}"}
                        ]
                        
                        return asyncio.run(self._get_response(consolidation_chat_history))
                    else:
                        return results[0]
                        
                except Exception as nested_e:
                    print(f"Error during split processing: {str(nested_e)}")
                    return f"Error: Failed to process text after attempts to split. {str(nested_e)}"
            else:
                print(f"Error during summarization: {str(e)}")
                return f"Error: {str(e)}"