import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def create_styled_table(df, title, companies_list, entity_column='entity_name', document_column='document_type'):
    """
    Create a styled table showing document distribution by company and document type
    
    Args:
        df: DataFrame containing the data
        title: Title for the table
        companies_list: List of company names to include
        entity_column: Column name for entities (default: 'entity_name')
        document_column: Column name for document types (default: 'document_type')
    """
    # Create pivot table
    pivot_table = df.groupby([entity_column, document_column])['document_id'].nunique().unstack(fill_value=0)
    pivot_table = pivot_table.reindex(companies_list, fill_value=0)
    normal_table = pivot_table.reset_index()
    normal_table.columns.values[0] = 'Company'

    n_rows = len(normal_table)
    row_height = 0.4
    fig_height = max(2, n_rows * row_height + 1.5)

    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=normal_table.values,
                     colLabels=normal_table.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Header styling
    for i in range(len(normal_table.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Row striping
    for i in range(1, len(normal_table) + 1):
        for j in range(len(normal_table.columns)):
            table[(i, j)].set_facecolor('#e0e0e0' if i % 2 == 0 else 'white')

    plt.figtext(0.5, 0.95, title, fontsize=16, fontweight='bold', ha='center')
    plt.show()


def prepare_data_report_0(df_by_theme, df_by_company_with_responses, user_selected_nb_topics_themes):
    """
    Prepare data for the regulatory report by ranking topics and companies
    
    Args:
        df_by_theme: DataFrame with theme-level data
        df_by_company_with_responses: DataFrame with company-level data and responses
        user_selected_nb_topics_themes: Number of top themes to select
    
    Returns:
        tuple: (top_by_theme, top_by_company) DataFrames
    """
    ### Section 1 - Sector-Wide Issues

    user_selected_ranking = ['theme', 'n_documents']
    user_selected_ascending_order = [True, False]

    top_by_theme = df_by_theme.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order).groupby(['theme'], as_index=False).head(user_selected_nb_topics_themes)
    top_by_theme = top_by_theme.reset_index(drop=True)

    ### Section 2 - Company-Specific Issues

    user_selected_nb_topics = 1
    user_selected_columns = ['entity_name', 'topic', 'headline', 'n_documents', 'response_summary', 'n_response_documents']

    list_tops_by_company = []

    user_selected_ranking = ['entity_name', 'n_documents']
    user_selected_ascending_order = [True, False]
    top_by_company = df_by_company_with_responses.copy()
    top_by_company['headline'] = top_by_company['topic_summary']
    top_by_company = top_by_company.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order).groupby(['entity_name'], as_index=False).head(user_selected_nb_topics)
    top_by_company = top_by_company[user_selected_columns]
    top_by_company['criterion'] = '1. Most Reported Issue'
    list_tops_by_company.append(top_by_company)

    user_selected_ranking = ['entity_name', 'risk_score', 'n_documents']
    user_selected_ascending_order = [True, False, False]
    top_by_company = df_by_company_with_responses.copy()
    top_by_company['headline'] = top_by_company['risk_summary']
    top_by_company = top_by_company.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order).groupby(['entity_name'], as_index=False).head(user_selected_nb_topics)
    top_by_company = top_by_company[user_selected_columns]
    top_by_company['criterion'] = '2. Biggest Risk'
    list_tops_by_company.append(top_by_company)

    user_selected_ranking = ['entity_name', 'uncertainty_score', 'n_documents']
    user_selected_ascending_order = [True, False, False]
    top_by_company = df_by_company_with_responses.copy()
    top_by_company['headline'] = top_by_company['uncertainty_explanation']
    top_by_company = top_by_company.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order).groupby(['entity_name'], as_index=False).head(user_selected_nb_topics)
    top_by_company = top_by_company[user_selected_columns]
    top_by_company['criterion'] = '3. Most Uncertain Issue'
    list_tops_by_company.append(top_by_company)

    top_by_company = pd.concat(list_tops_by_company)
    top_by_company = top_by_company[user_selected_columns+['criterion']]
    top_by_company = top_by_company.sort_values(by=['entity_name', 'criterion'])
    top_by_company = top_by_company.reset_index(drop=True)

    return top_by_theme, top_by_company


def generate_html_report(df_theme, df_entities, title):
    """
    Generate an HTML report from theme and entity dataframes
    
    Args:
        df_theme: DataFrame with theme-level data
        df_entities: DataFrame with entity-level data
        title: Report title
    
    Returns:
        str: Complete HTML report as string
    """
    # Generate current report date
    report_date = datetime.now().strftime("%B %d, %Y")

    # Section 1: Themes, Topics, and Summaries
    theme_boxes = ""

    for theme in df_theme['theme'].unique():
        theme_summary = df_theme[df_theme['theme'] == theme]
        topics_html = "".join(
            [f"<li><strong>{row['topic']}</strong>: {row['topic_summary']}</li>" for _, row in theme_summary.iterrows()]
        )
        theme_boxes += f"""
        <div class='report-theme-box'>
            <h3>{theme}</h3>
            <ul>{topics_html}</ul>
        </div>
        """

    # Section 2: Headlines by Entity (Horizontal layout)
    headline_sections = ""
    entity_groups = df_entities.groupby('entity_name')

    for entity, group in entity_groups:
        headline_sections += f"<div class='report-entity'>"
        headline_sections += f"<h3>{entity}</h3>"
        headline_sections += "<div class='report-flex-container'>"

        for _, row in group.iterrows():
            headline_sections += f"<div class='report-criterion-box'>"
            headline_sections += f"<strong class='report-criterion'>{row['criterion']}</strong><br/>"
            headline_sections += f"<strong class='topic'>{row['topic']}:</strong> {row['headline']}<br>[{row['n_documents']} News]<br/>"
            headline_sections += "</div>"

        headline_sections += "</div>"  # Close flex container
        headline_sections += "<br/>"  # Blank line

        response_summary_items = group[['topic', 'response_summary']].dropna().drop_duplicates()
        if response_summary_items.size > 0:
            headline_sections += "<div class='report-response-summary'>"
            headline_sections += f"<strong>Company's Response:</strong><br/>"
            headline_sections += "<ul>"
            for _, row in response_summary_items.iterrows():
                headline_sections += f"<li><strong>{row['topic']}</strong>: {row['response_summary']}</li>"
            headline_sections += "</ul></div>"

        headline_sections += "</div>"  # Close entity div

    # Complete HTML structure
    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            .report-container {{
                font-family: Arial, sans-serif;
                padding: 30px;
                line-height: 1.6;
                background-color: #ffffff;
                color: #333;
            }}
            .report-container h1 {{
                color: #003A70;
                font-size: 24px;
                margin-bottom: 5px;
                font-weight: 700;
                border-bottom: 2px solid #003A70;
                padding-bottom: 10px;
                text-align: center;
            }}
            .report-date {{
                font-size: 16px;
                color: #555;
                margin-bottom: 20px;
                text-align: center;
            }}
            .report-container h2 {{
                color: #003A70;
                font-size: 20px;
                margin-top: 30px;
                font-weight: 600;
            }}
            .report-theme-box, .report-entity {{
                border: 2px solid #003A70;
                margin: 25px;
                padding: 20px;
                border-radius: 8px;
                background: #F7F9FC;
            }}
            .report-flex-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                gap: 15px;
            }}
            .report-criterion-box {{
                flex: 1;
                min-width: 200px;
                padding: 15px;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
                background: #FFFFFF;
            }}
            .report-criterion {{
                display: inline-block;
                padding: 5px;
                background-color: #003A70;
                color: white;
                border-radius: 5px;
                margin-bottom: 5px;
                font-size: 14px;
            }}
            .report-response-summary {{
                padding: 15px;
                border: 1px solid #B0B0B0;
                border-radius: 5px;
                background: #FFFFFF;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h1>{title}</h1>
            <div class="report-date">{report_date}</div>

            <h2>Sector-Wide Issues</h2>
            {theme_boxes}

            <h2>Company-Specific Issues</h2>
            {headline_sections}
        </div>
    </body>
    </html>
    """

    return html_report