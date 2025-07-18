from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def generate_html_report_v4(df, title, definitions, section_title):

    # Generate current report date.
    report_date = datetime.now().strftime("%B %d, %Y")

    def build_entity_section(df):
        """
        Build the HTML for each row in the dataframe.
        Expects each row to contain:
        - entity_name
        - risk_summary
        - proactivity_summary
        - ai_proactivity_minus_disruption_risk_score
        - ai_disruption_risk_score
        - ai_proactivity_score
        - n_documents_risk
        - n_documents_proactivity
        """
        sections = ""
        for _, record in df.iterrows():
            entity = record["entity_name"]
            risk_summary = record["risk_summary"] if pd.notnull(record["risk_summary"]) else "No relevant content was retrieved."
            proactivity_summary = record["proactivity_summary"] if pd.notnull(record["proactivity_summary"]) else "No relevant content was retrieved."

            ai_proactivity_minus_disruption_risk_score = f"{record['ai_proactivity_minus_disruption_risk_score']:.2f}" if pd.notnull(record["ai_proactivity_minus_disruption_risk_score"]) else "N/A"
            ai_disruption_risk_score = f"{record['ai_disruption_risk_score']:.2f}" if pd.notnull(record["ai_disruption_risk_score"]) else "N/A"
            ai_proactivity_score = f"{record['ai_proactivity_score']:.2f}" if pd.notnull(record["ai_proactivity_score"]) else "N/A"

            nb_documents_risk = f"{record['n_documents_risk']}" if pd.notnull(record["n_documents_risk"]) else "N/A"
            nb_documents_proactivity = f"{record['n_documents_proactivity']}" if pd.notnull(record["n_documents_proactivity"]) else "N/A"

            # Build header HTML: display the entity name.
            header_html = f"<h3>{entity}</h3>"

            # Build a dedicated score box with scores in three rows.
            score_box_html = f"""
                <div class="report-score-box">
                    <div class="score-row">
                        <p><strong>AI Proactivity Minus Disruption Risk Score:</strong> {ai_proactivity_minus_disruption_risk_score}</p>
                    </div>
                    <div class="score-row">
                    <p>
                        <strong>AI Disruption Risk Score:</strong> {ai_disruption_risk_score} / 
                        <strong>Nb Documents Risk:</strong> {nb_documents_risk}
                    </p>
                </div>
                <div class="score-row">
                    <p>
                        <strong>AI Proactivity Score:</strong> {ai_proactivity_score} / 
                        <strong>Nb Documents Proactivity:</strong> {nb_documents_proactivity}
                    </p>
                    </div>
                </div>
            """

            # Build the summary boxes (Risk Summary and Proactivity Summary) as before.
            summary_boxes = f"""
                <div class="report-flex-container">
                    <div class="report-criterion-box">
                        <h4>AI Disruption Risk</h4>
                        <p>{risk_summary}</p>
                    </div>
                    <div class="report-criterion-box">
                        <h4>AI Proactivity</h4>
                        <p>{proactivity_summary}</p>
                    </div>
                </div>
            """

            sections += "<div class='report-entity'>"
            sections += header_html
            sections += score_box_html
            sections += summary_boxes
            sections += "</div>"  # Close entity block
        return sections

    # Build company section using the provided df.
    company_section = build_entity_section(df)

    # Build definitions section by iterating over the dictionary.
    definitions_html = ""
    for score, definition in definitions.items():
        definitions_html += f"<p><strong>{score}</strong>: {definition}</p>"

    definitions_section = f"""
        <div class="report-section-box">
            <p class="report-section-title">Score Definitions</p>
            {definitions_html}
        </div>
    """

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
                background-color: #ffffff;
                color: #333;
            }}
            .report-container h1 {{
                color: #003A70;
                font-size: 24px;
                margin-bottom: 5px;
                font-weight: 700;
                text-align: center;
            }}
            .report-date {{
                font-size: 16px;
                color: #555;
                margin-bottom: 20px;
                text-align: center;
            }}
            .report-section-box {{
                border: 1px solid #003A70;
                padding: 15px;
                margin: 25px 0;
                border-radius: 8px;
                background: #FAFBFC;
            }}
            .report-section-title {{
                font-size: 22px;
                color: #003A70;
                margin: 0 0 15px 0;
                text-align: left;
            }}
            .report-entity {{
                border: 2px solid #003A70;
                margin: 15px 0;
                padding: 20px;
                border-radius: 8px;
                background: #F7F9FC;
            }}
            .report-score-box {{
                border: 1px solid #B0B0B0;
                padding: 10px;
                margin: 10px 0;
                background: #FFFFFF;
                border-radius: 5px;
            }}
            .score-row {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .score-row p {{
                margin: 5px 0;
                font-size: 14px;
                color: #333;
            }}
            .entity-subheader {{
                font-size: 16px;
                font-weight: bold;
                color: grey;
                margin-top: -5px;
                margin-bottom: 20px;
            }}
            .report-flex-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                justify-content: space-between;
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
            h3 {{
                margin-bottom: 5px;
                font-size: 20px;
                color: #003A70;
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
            <h1>{title}</h1>
            <div class="report-date">{report_date}</div>

            {definitions_section}

            <div class="report-section-box">
                <p class="report-section-title">{section_title}</p>
                {company_section}
            </div>
        </div>
    </body>
    </html>
    """
    return html_report


def display_report(report, score, top, nb_entities, export_to_path=None):

    dict_definitions = {
            'AI Disruption Risk Score' : 'The number of unique documents related to the risk of GenAI disruption, normalized by the average number of risk documents retrieved across the entire watchlist.',
            'AI Proactivity Score' : 'The number of unique documents related to the proactive adoption of GenAI, normalized by the average number of proactivity documents retrieved across the entire watchlist.',
            'AI Proactivity Minus Disruption Risk Score' : "The difference between the AI Proactivity Score and the AI Disruption Risk Score. A higher score indicates a stronger company response relative to the identified risks.",
            }
    
    score_name_dict = {
            'ai_disruption_risk_score' : 'AI Disruption Risk Score',
            'ai_proactivity_score' : 'AI Proactivity Score',
            'ai_proactivity_minus_disruption_risk_score' : 'AI Proactivity Minus Disruption Risk Score',
    }
    
    if top == 'top':
        df = report.report_by_company.sort_values(by=[score], ascending=[False]).head(nb_entities).copy()
        section_title = 'Top ' + str(nb_entities) + ' Companies for ' + score_name_dict[score]
    elif top == 'bottom':
        df = report.report_by_company.sort_values(by=[score], ascending=[True]).head(nb_entities).copy()
        section_title = 'Bottom ' + str(nb_entities) + ' Companies for ' + score_name_dict[score]
    elif top == 'all':
        df = report.report_by_company.sort_values(by=['entity_name'], ascending=[True]).copy()
        section_title = ''
    df = df.reset_index(drop=True)

    html_content = generate_html_report_v4(df=df, title='AI Disruption Risk and Proactive Responses', definitions=dict_definitions, section_title=section_title)

    if export_to_path:
        # save the html report
        with open(export_to_path, 'w') as file:
            file.write(html_content)

    display(HTML(html_content)) 


def plot_company_scores(df, score_1, score_2, title):
    """
    Plots entity_name on a plane defined by score_1 and score_2.

    Parameters:
    df (pd.DataFrame): DataFrame containing score_1, score_2, and 'entity_name' columns.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot: each point represents a company
    ax.scatter(df[score_1], df[score_2], color='blue', alpha=0.6)

    # Add the company name as a label on each point
    for _, row in df.iterrows():
        ax.annotate(row['entity_name'],
                    (row[score_1], row[score_2]),
                    textcoords="offset points",  # Use offset to position text more cleanly
                    xytext=(5, 5),               # Offset: 5 points right and 5 points up
                    ha='left',                   # Horizontal alignment of text
                    fontsize=9)                  # Adjust font size as needed

    # Set plot labels and title
    ax.set_xlabel('AI Disruption Risk Score')
    ax.set_ylabel('AI Proactivity Score')
    ax.set_title(title)

    # Optionally, adjust grid or styling
    ax.grid(True)

    # Show the plot
    plt.show()


def plot_company_scores_log_log(df, score_1, score_2, title):
    """
    Plots entity_name on a plane defined by score_1 and score_2.

    Parameters:
    df (pd.DataFrame): DataFrame containing score_1, score_2, and 'entity_name' columns.
    """

    df_log = df.copy()
    df_log[score_1] = np.log(df_log[score_1].clip(lower=0.01))
    df_log[score_1] = df_log[score_1] + np.abs(df_log[score_1].min())
    df_log[score_2] = np.log(df_log[score_2].clip(lower=0.01))
    df_log[score_2] = df_log[score_2] + np.abs(df_log[score_2].min())

    plot_company_scores(df_log, score_1, score_2, title + ' - log-log scale')

