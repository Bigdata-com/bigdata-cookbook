import pandas as pd 

def generate_prompt_template() -> str:
    """
    Returns the base prompt template with placeholders for formatting.
    """
    return """
    You are an expert financial analyst with specialized knowledge in thematic investment research.
    Your task is to generate a concise motivation statement explaining why this company is included in a thematic watchlist.

    Theme: {theme}
    Company: {company}

    This company has {total_quotes} quotes related to the theme, with exposure to the following sub-themes:
    {label_summary}

    Here are the quotes with their corresponding labels:
    {quotes_and_labels}

    Generate a concise motivation statement (2-4 sentences) that:
    1. ALWAYS begins with the company name
    2. Summarizes WHY this company is included in the thematic watchlist
    3. References the specific sub-themes (from Label column) where the company shows strongest exposure (has the most number of elements in 'Quote' column)
    4. For any numerical figures, make sure to quote the exact metric correctly
    5. Uses objective, evidence-based language referring to the company's actual activities
    6. Maintains a neutral, analytical tone without subjective judgments
    7. Focuses on facts rather than predictions or recommendations
    8. Keeps the statement concise ({min_words}-{max_words} words)
    """

def get_motivation_prompt(company: str, data: pd.DataFrame, theme_name: str, min_words: int, max_words: int) -> str:
    """
    Formats the motivation prompt using company data and the prompt template.

    Parameters:
    - company (str): Company name
    - data (dict): Dictionary with 'label_counts', 'quotes_and_labels', and 'total_quotes'
    - theme_name (str): Name of the theme
    - min_words (int): Minimum word count
    - max_words (int): Maximum word count

    Returns:
    - str: Fully formatted motivation prompt
    """
    label_summary = "\n".join([f"- {label}: {count} quotes" for label, count in data['label_counts']])

    quotes_text = ""
    for i, item in enumerate(data['quotes_and_labels']):
        quotes_text += f"{i+1}. \"{item['quote']}\" [Label: {item['label']}]\n"

    prompt_template = generate_prompt_template()

    return prompt_template.format(
        theme=theme_name,
        company=company,
        total_quotes=data['total_quotes'],
        label_summary=label_summary,
        quotes_and_labels=quotes_text,
        min_words=min_words,
        max_words=max_words
    )

