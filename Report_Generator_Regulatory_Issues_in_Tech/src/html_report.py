from datetime import datetime
import pandas as pd

def prepare_data_report_0(df_by_theme, df_by_company_with_responses, user_selected_nb_topics_themes):

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


def prepare_data_report_1(df_by_theme, df_by_company_with_responses, nb_bullet_points):
    ### Section 1 - Sector-Wide Issues
    
    user_selected_ranking = ['theme', 'n_documents']
    user_selected_ascending_order = [True, False]
    user_selected_nb_topics = 10
    
    top_by_theme = df_by_theme.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order).groupby(['theme'], as_index=False).head(user_selected_nb_topics)
    top_by_theme = top_by_theme.reset_index(drop=True)
    
    
    ### Section 2 - Company-Specific Issues
    
    user_selected_nb_topics_per_company = 1
    user_selected_nb_topics_total = nb_bullet_points
    user_selected_columns = ['entity_name', 'topic', 'headline', 'n_documents', 'response_summary', 'n_response_documents']
    
    user_selected_ranking = ['risk_score', 'uncertainty_score', 'n_documents']  
    user_selected_ascending_order = [False, False, False]
    top_by_company = df_by_company_with_responses.copy()
    top_by_company['headline'] = top_by_company['topic_summary']
    top_by_company = top_by_company.sort_values(by=['entity_name']+user_selected_ranking, ascending=[True]+user_selected_ascending_order).groupby(['entity_name'], as_index=False).head(user_selected_nb_topics_per_company)
    top_by_company = top_by_company.sort_values(by=user_selected_ranking, ascending=user_selected_ascending_order)
    top_by_company = top_by_company[user_selected_columns]
    top_by_company = top_by_company.head(user_selected_nb_topics_total)

    return top_by_theme, top_by_company


def generate_html_report(df_theme, df_entities, title):
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


def generate_html_report_v1(df_theme, df_entities, title):

   # Generate current report date
    report_date = datetime.now().strftime("%B %d, %Y")

    # Section 1: Universe‑Wide (Themes, Topics, Summaries, and Document Counts)
    theme_boxes = ""
    for theme in df_theme['theme'].unique():
        theme_subset = df_theme[df_theme['theme'] == theme]
        topics_html = "".join(
            [f"<li><strong>{row['topic']}</strong>: {row['topic_summary']}<br/>"
             f"[{row['n_documents']} Documents]</li>" 
             for _, row in theme_subset.iterrows()]
        )
        theme_boxes += f"""
        <div class='report-theme-box'>
            <h3>{theme}</h3>
            <ul>{topics_html}</ul>
        </div>
        """

    # Section 2: Company‑Specific (Each entity gets exactly 2 horizontal boxes)
    # Group by 'entity_name' with sort=False to preserve original order
    entity_sections = ""
    for entity, group in df_entities.groupby('entity_name', sort=False):
        # Select the record with the maximum number of news documents for "Most Reported Topic"
        most_reported_row = group.loc[group['n_documents'].idxmax()]

        entity_sections += f"<div class='report-entity'>"
        entity_sections += f"<h3>{entity}</h3>"
        entity_sections += "<div class='report-flex-container'>"
        
        # Box 1: Most Reported Topic
        entity_sections += "<div class='report-criterion-box'>"
        entity_sections += "<h4>Most Reported Topic</h4>"
        entity_sections += f"<p><strong>{most_reported_row['topic']}</strong>: {most_reported_row['headline']}<br/>"
        entity_sections += f"[{most_reported_row['n_documents']} News]</p>"
        entity_sections += "</div>"  # Close Most Reported Topic box
        
        # Box 2: Company's Response
        entity_sections += "<div class='report-criterion-box'>"
        entity_sections += "<h4>Company's Response</h4>"
        if pd.notnull(most_reported_row['response_summary']):
            entity_sections += f"<p>{most_reported_row['response_summary']}<br/>"
            # entity_sections += f"[{most_reported_row['n_response_documents']} Responses]</p>"
        else:
            entity_sections += "<p>No response provided</p>"
        entity_sections += "</div>"  # Close Company's Response box
        
        entity_sections += "</div>"  # Close flex container
        entity_sections += "</div>"  # Close entity div

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
            .report-criterion-box h4 {{
                margin-top: 0;
                color: #003A70;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h1>{title}</h1>
            <div class="report-date">{report_date}</div>

            <h2>Universe-Wide</h2>
            {theme_boxes}

            <h2>Company-Specific (Top {len(df_entities)} Risks)</h2>
            {entity_sections}
        </div>
    </body>
    </html>
    """

    return html_report