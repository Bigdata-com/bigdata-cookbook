from datetime import datetime
import os
import os
import jinja2
from IPython.core.display import HTML
import pandas as pd
import unicodedata
import re
from typing import Optional


def clean_text(text):
    # Check if the text is a string, otherwise return it as-is (to handle NaN or non-string values)
    if not isinstance(text, str):
        return text
    
    # Normalize the text to remove any weird encodings
    text = unicodedata.normalize('NFKD', text)
    
    # Check if the dollar sign has already been replaced, and only replace if it hasn't
    if not r'\$' in text:
        text = text.replace('$', r'\$')
    
    # Remove any unintended italic or mathematical symbols by replacing them
    text = re.sub(r'[\u2061-\u2064\u0338-\u0339\u2212-\u2213\u200E-\u200F]', '', text)
    
    # Ensure spacing is correct by replacing multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text


# Function to generate a report for a single date
def generate_html_report(date, day_in_review, topics, main_theme, template_path="./report_template.html"):
    # Load the Jinja2 template
    template_loader = jinja2.FileSystemLoader(searchpath=f"{os.getcwd()}/assets/")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_path)

    # Generate the title based on the main theme
    title = f"{main_theme}"
    
    # Render the template with data
    html_output = template.render(
        date=date,
        day_in_review=day_in_review,
        topics=topics,
        main_theme=title  # Pass the dynamic title
    )
    
    # Replace asset references with embedded base64 images
    assets_dir = f"{os.getcwd()}/assets/"
    asset_files = {
        'assets/bigdata-logo-white.svg': 'bigdata-logo-white.svg',
        'assets/flame-icon.png': 'flame-icon.png',
        'assets/arrow_up.png': 'arrow_up.png',
        'assets/arrow_mid.png': 'arrow_mid.png',
        'assets/arrow_down.png': 'arrow_down.png'
    }
    
    for asset_path, asset_file in asset_files.items():
        if os.path.exists(os.path.join(assets_dir, asset_file)):
            with open(os.path.join(assets_dir, asset_file), "rb") as file:
                import base64
                file_content = base64.b64encode(file.read()).decode('utf-8')
                file_extension = os.path.splitext(asset_file)[1].lower()
                mime_type = "image/svg+xml" if file_extension == '.svg' else f"image/{file_extension[1:]}"
                data_url = f"data:{mime_type};base64,{file_content}"
                html_output = html_output.replace(f'src="{asset_path}"', f'src="{data_url}"')
    
    return html_output

def save_html_report(html_output, report_date, theme, output_base_dir=None):
    # Define the output directory and file name
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
        output_file = os.path.join(output_base_dir, f"{report_date}_{theme.replace(' ', '_')}.html")
    else:
        os.makedirs('report', exist_ok=True)
        output_file = f"./report/{report_date}_{theme.replace(' ', '_')}.html"

    # Save the HTML output to a file
    with open(output_file, "w") as f:
        f.write(html_output)

    print(f"Report saved to {output_file}")

# Helper functions for sorting
def novelty_score_value(novelty_score):
    score_map = {'Novel': 3, 'Moderate': 2, 'Repeat': 1}
    return score_map.get(novelty_score, 0)  # Default to 0 if novelty_score is not recognized

def magnitude_value(magnitude):
    score_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Neutral': 0}
    return score_map.get(magnitude, 0)  # Default to 0 if magnitude is not recognized

def number_of_news_value(number_of_news):
    return number_of_news  # Use directly for sorting

def prepare_data_for_report(df, ranking_criteria, report_date: Optional[str] = None, impact_filter: Optional[str] = None):
    # Applying the cleaning function to the text in your DataFrame before rendering
    df['Summary'] = df['Summary'].apply(clean_text)
    df['Day_in_Review'] = df['Day_in_Review'].apply(clean_text)
    df['Text_Summary'] = df['Text_Summary'].apply(clean_text)
    df['Topic'] = df['Topic'].apply(clean_text)
    
    # Define the mapping for novelty scores
    novelty_mapping = {
        'New': 'Novel',
        'Moderate': 'Moderate',
        'Old': 'Repeat'
    }
    
    # Remove rows where 'Source' or 'Text_Summary' is NaN or empty
    
    df = df[(df['Source'].str.strip() != '') & (df['Text_Summary'].str.strip() != '')]

    # Apply impact filter if provided
    if impact_filter is not None:
        if impact_filter == 'positive_impact':
            df = df[df['Impact_Score'].str.lower() == 'positive']
        elif impact_filter == 'negative_impact':
            df = df[df['Impact_Score'].str.lower() == 'negative']

    reports = []
    criteria_map = {
        'novelty': lambda x: novelty_score_value(novelty_mapping.get(x['novelty_score'], x['novelty_score'])),
        'magnitude': lambda x: magnitude_value(x['magnitude']),
        'volume': lambda x: number_of_news_value(x['number_of_news'])
    }

    # Convert report_date to datetime.date if provided
    if report_date is not None:
        try:
            report_date = datetime.strptime(report_date, "%Y-%m-%d").date()
            # Filter DataFrame by date
            df = df[df['Date'] == report_date]
        except ValueError:
            print(f"Invalid date format: {report_date}. Expected format is YYYY-MM-DD.")
            return []

    grouped = df.groupby('Date')

    def parse_day_in_review(text):
            bullets = re.split(r'\s*-\s+', text)
            parsed = []
            for bullet in bullets:
                bullet = bullet.strip()
                if not bullet:
                    continue
                # Wrap leading **...**: in <b>...</b> if present
                bullet = re.sub(r'^\*\*(.+?)\*\*:', r'<b>\1</b>:', bullet)
                parsed.append(bullet.replace(r'\$', '$'))
            return parsed

    for date, group in grouped:
        topics = []
        top_topics = group.groupby('Topic').agg({
            'Summary': 'first',
            'Day_in_Review': 'first',
            'Novelty_Score': 'first',
            'Magnitude_Score': 'first',
            'Impact_Score': 'first'
        }).reset_index()
        
        day_in_review = parse_day_in_review(top_topics['Day_in_Review'].iloc[0]) if 'Day_in_Review' in top_topics.columns else []

        for _, topic_row in top_topics.iterrows():
            topic = topic_row['Topic']
            summary = topic_row['Summary']
            # Apply the novelty mapping
            novelty_score = novelty_mapping.get(topic_row['Novelty_Score'], topic_row['Novelty_Score'])
            magnitude = topic_row['Magnitude_Score']
            impact = topic_row['Impact_Score']
            
            # Filter out rows with empty 'Source' or 'Text_Summary'
            associated_news = group[group['Topic'] == topic].dropna(subset=['Source', 'Text_Summary'])
            associated_news = associated_news[(associated_news['Source'].str.strip() != '') & (associated_news['Text_Summary'].str.strip() != '')]
            
            sources_and_summaries = associated_news[['Source', 'Text_Summary']].values.tolist()
            
            # Calculate the number of news items after cleaning
            number_of_news = len(associated_news)
            
            topics.append({
                'topic': topic,
                'summary': summary,
                'novelty_score': novelty_score,
                'magnitude': magnitude,
                'impact': impact,
                'sources_and_summaries': sources_and_summaries,
                'number_of_news': number_of_news
            })
        
        # Sort topics based on the provided ranking criteria
        topics = sorted(topics, key=lambda x: tuple(criteria_map[crit](x) for crit in ranking_criteria), reverse=True)
        
        reports.append({
            'date': date.strftime('%Y-%m-%d'),
            'day_in_review': day_in_review,
            'topics': topics
        })

    return reports