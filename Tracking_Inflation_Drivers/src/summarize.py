from src.token_manager import TokenManager
from datetime import datetime
from openai import OpenAI
import os
import pandas as pd

def topic_summary(text_to_analyze, model, number_of_reports, api_key, main_theme, topic, start_date, end_date):
    # Create the prompt incorporating the topic
    prompt = (
        f""""You are an expert macroeconomic analyst tasked with summarizing the key aspects of {topic} related to {main_theme} based on a set of daily news reports.

        Your goal is to analyze the provided text and summarize the key elements related to {topic} in the context of {main_theme}.

    Text Description:
    - The input text includes {number_of_reports} market reports, each explicitly structured with its Headline and Text. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
    - **Summarize the text:**  Provide an **analytical summary** of the key mentioned aspects of {topic} in the context of {main_theme} across all reports. The summary should have at least 3 sentences.
    - **High Level of Detail:** The **analytical summary should be rich with facts and figures and contain as many insights as possible.
    - **Reporting Dates**: Focus on facts and figures associated with periods between {start_date} and {end_date}. Disregard anything older than {start_date}. Report relevant reporting dates between {start_date} and {end_date}.
    
    Output:
    **Summary:** A summary of the key insights related to {topic} in the context of {main_theme}

    Format:
    Return the result as a JSON object with the following structure:
    {{ "summary": "Your summary here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    try:
        summary = eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary = ''
    
    return summary


def summarize_topic_by_theme(df_labeled, list_specific_themes,sub_topics, themes_tree_dict, api, start_date, end_date):

    df_topics_themes = df_labeled.loc[~df_labeled.label.isin(['','Not Applicable', 'unassigned', 'unclear'])].copy()
    df_topics_themes = df_topics_themes.rename(columns={'Headline': 'headline', 'Quote': 'text', 'Document ID': 'rp_document_id'})
    model = 'gpt-4o-mini-2024-07-18'
    api_key = api

    list_df_by_theme = []
    
    for main_theme in list_specific_themes:

        for topic in sub_topics:

            # print('topic = ' + topic)
            # sentence_topic = get_label_dict_from_tree(themes_tree_dict[main_theme])[topic]
            
            df_to_summarize = df_topics_themes.loc[(df_topics_themes.theme==main_theme)&(df_topics_themes.label==topic)]

            n_documents = df_to_summarize.rp_document_id.nunique()
            # print('n_documents = ' + str(n_documents))
            
            df_text_to_summarize = df_to_summarize[['headline', 'text']].drop_duplicates()
            df_text_to_summarize['text_to_summarize'] = (
                    '--- Report Start ---\n'     
                ' Headline: ' + df_text_to_summarize.headline.astype(str) + 
                    '\n Text: ' + df_text_to_summarize.text.astype(str) +
                    '\n --- Report End ---'
                )
            n_reports = df_text_to_summarize.shape[0]

            if n_reports==0:
                continue
            
            token_manager = TokenManager(
                model_name="gpt-4o-mini-2024-07-18",
                max_tokens=100000,
                verbose=False
            )

            list_text_to_summarize = token_manager.split_text_by_chunks(df_text_to_summarize)
            
            if len(list_text_to_summarize)>1:
                list_summary = []
                for this_text in list_text_to_summarize:
                    this_summary = topic_summary(this_text, model, n_reports, api_key, main_theme, topic, start_date, end_date)
                    this_summary = (
                            '--- Report Start ---\n'
                            ' Headline: ' + '' + 
                            '\n Text: ' + this_summary +
                            '\n --- Report End ---'
                        )
                    list_summary.append(this_summary)
                text_to_summarize = "\n\n".join(list_summary)                    
            else:
                text_to_summarize = list_text_to_summarize[0]

            summary = topic_summary(text_to_summarize, model, n_reports, api_key, main_theme, topic, start_date, end_date)

            df_by_theme = pd.DataFrame({'theme': [main_theme], 'topic': [topic], 'topic_summary': [summary], 'n_documents': [n_documents]})
            list_df_by_theme.append(df_by_theme)

    df_by_theme = pd.concat(list_df_by_theme)
    df_by_theme = df_by_theme.reset_index(drop=True)
    
    return df_by_theme

## Create Intro Section
def call_llm_intro_section(text_to_analyze, model, api_key, main_theme):
    # Create the prompt incorporating the topic
    prompt = (
        f""""You are an expert macroeconomic analyst tasked with summarizing the key pieces of information related to {main_theme}.

        Your goal is to read detailed reports by topic in the provided text and summarize the key elements related to {main_theme} providing an overview of the most important contents.

    Text Description:
    - The input text includes detailed summaries on specific topics related to {main_theme}, each explicitly structured with its Topic and Sub-Topic tags. Each report is clearly separated by a '--- Report End ---' marker.

    Instructions:
    **Summarize the text:**  Provide a **short, synthetic summary** of the key mentioned aspects in the context of {main_theme} across all reports. The summary should have between 3 and 5 sentences.
    
    Output:
    **Report overviews:** A summary of the key aspects related to {main_theme}

    Format:
    Return the result as a JSON object with the following structure:
    {{ "summary": "Your summary here"}}
    """
    )

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Send request to OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": text_to_analyze
            }
        ],
        response_format={ "type":"json_object"},
        temperature=0
    )
    
    # Extract and process the response
    summary = response.model_dump()

    try:
        summary = eval(summary['choices'][0]['message']['content'])['summary']
    except:
        summary = ''
    
    return summary

def create_intro_section(df_labeled, api, main_theme):
    
    model = 'gpt-4o-mini-2024-07-18'
    api_key = api

    list_df_by_theme = []
    
    for i, row in df_labeled.iterrows():
        df_labeled['text_to_summarize'] = (
                    '--- Report Start ---\n'
                    ' Topic: ' + df_labeled.theme.astype(str) +
                    ' Sub-Topic: ' + df_labeled.topic.astype(str) +
                    '\n Text: ' + df_labeled.topic_summary.astype(str) +
                    '\n --- Report End ---'
                )
    token_manager = TokenManager(
                model_name="gpt-4o-mini-2024-07-18",
                max_tokens=100000,
                verbose=False
            )

    list_text_to_summarize = token_manager.split_text_by_chunks(df_labeled)
    if len(list_text_to_summarize)>1:
        list_summary = []
        for this_text in list_text_to_summarize:
            this_summary = call_llm_intro_section(this_text, model, api, main_theme)
            this_summary = (
                            '--- Report Start ---\n'
                            '\n Text: ' + this_summary +
                            '\n --- Report End ---'
                        )
            list_summary.append(this_summary)
            text_to_summarize = "\n\n".join(list_summary)                    
    else:
        text_to_summarize = list_text_to_summarize[0]

    summary = call_llm_intro_section(text_to_summarize, model, api, main_theme)
    
    return summary

# HTML visualization function
def create_html_report(main_theme, start_date, end_date, intro_section, df):
    if df.empty:  # Check if df is empty
        return "<p>No data available to display.</p>"
    
    # Group the dataframe by 'Theme'
    grouped = df.groupby('label')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{main_theme}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
                background-color: white; /* Set the background color to white */
                color: #333; /* Text color */
            }}
            h1 {{
                color: #333;
            }}
            h2 {{
                color: #555;
            }}
            p {{
                margin: 10px 0;
            }}
            .label {{
                font-weight: bold;
                color: white; /* Text color for the label */
                background-color: blue; /* Blue background for the label */
                padding: 5px; /* Add some padding */
                border-radius: 3px; /* Rounded corners */
                display: inline-block; /* Make the label fit content */
            }}
            .theme {{
                margin-top: 30px;
                border-top: 2px solid #ccc;
                padding-top: 20px;
                padding-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{main_theme} Report</h1>
        <p>Report from {start_date} to {end_date}</p>
        <p>{intro_section}</p>
        
        {''.join(
            f'<h2>{label.replace("factors", "").title()} Factors</h2>' +
            ''.join(
                f'<p class="topic-summary"><strong>{row["theme"]}: {row["topic"]}.</strong> {row["topic_summary"]} </p>'
                for _, row in group.iterrows()
            )
            for label, group in grouped
        )}
    </body>
    </html>
    """
    return html_content

#f'<span class="label">{row["label"].replace("$", "&#36;")}</span>

def prepare_html_for_display(html):
    """
    Returns a version of the HTML string with dollar signs escaped
    so that MathJax won't process them in Jupyter displays.
    """
    return html.replace("$", "\$")