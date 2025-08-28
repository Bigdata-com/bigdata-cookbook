from datetime import datetime
import os
import jinja2
from IPython.core.display import HTML
import pandas as pd
import unicodedata
import re
from typing import Optional
from IPython.display import display, HTML, Image
import io
from IPython.display import Image, display
import shutil
import subprocess
import sys

def clean_text(text):
    # Check if the text is a string, otherwise return it as-is (to handle NaN or non-string values)
    if not isinstance(text, str):
        return text
    
    # Normalize the text to remove any weird encodings
    text = unicodedata.normalize('NFKD', text)
    
    # # Check if the dollar sign has already been replaced, and only replace if it hasn't
    # if not r'\$' in text:
    #     text = text.replace('$', r'\$')
    
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

def load_sanitize_display_html_report(report_date, theme, report_dir='./report/' ):
    """
    Dynamically constructs the path to the HTML report, loads, sanitizes, and displays it.
    - report_dir: directory where reports are saved (e.g., './report')
    - report_date: date string, e.g., '2025-06-25'
    - theme: theme string, e.g., 'Crude_Oil'
    """
    filename = f"{report_date}_{theme.replace(' ', '_')}.html"
    html_file_path = os.path.join(report_dir, filename)

    if not os.path.exists(html_file_path):
        print(f"Report file not found: {html_file_path}")
        return

    with open(html_file_path, "r") as f:
        html_content = f.read()

    # Sanitize for notebook display
    sanitized_html = re.sub(r'<!DOCTYPE html>', '', html_content, flags=re.IGNORECASE)
    sanitized_html = re.sub(r'</?(html|head|body)[^>]*>', '', sanitized_html, flags=re.IGNORECASE)

    display(HTML(sanitized_html))

def silent_html2image(html_path, output_path, browser_executable='/usr/bin/chromium'):
    """
    Runs html2image screenshot in a subprocess, suppressing all output.
    """
    script = f"""
import sys
from html2image import Html2Image
hti = Html2Image(browser_executable='{browser_executable}', custom_flags=['--no-sandbox', '--disable-gpu', '--disable-software-rasterizer'])
hti.output_path = f'{output_path}'
hti.screenshot(html_file='{html_path}', size=(1600, 3000), save_as='tmp_screenshot.png')
"""
    subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def display_html_report_as_image(report_date, theme, report_dir='./report/', tmp_dir='./output'):
    """
    Converts a saved HTML report to a PNG image and displays it in the notebook, without saving to disk.
    Captures the full page and hides scrollbars for a clean image.
    If Chromium is not found, falls back to displaying the HTML directly.
    Requires 'html2image' and 'Pillow' packages.
    """

    html_filename = f"{report_date}_{theme.replace(' ', '_')}.html"
    html_path = os.path.join(report_dir, html_filename)

    # Inject CSS to hide scrollbars and set overflow
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    custom_css = """
    <style>
    html, body { overflow: hidden !important; }
    ::-webkit-scrollbar { display: none; }
    </style>
    """
    if "<head>" in html_content:
        html_content = html_content.replace("<head>", f"<head>{custom_css}")
    else:
        html_content = custom_css + html_content
    temp_html_path = os.path.join(tmp_dir, f"temp_{html_filename}")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    chromium_path = shutil.which('chromium') or shutil.which('chromium-browser') or '/usr/bin/chromium'
    if not os.path.exists(chromium_path):
        display(HTML(html_content))
        os.remove(temp_html_path)
        return

    # Suppress stderr during screenshot
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        silent_html2image(temp_html_path, tmp_dir, browser_executable=chromium_path)
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr
    img_path = os.path.join(tmp_dir, "tmp_screenshot.png")
    display(Image(filename=img_path))
    os.remove(temp_html_path)

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