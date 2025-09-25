def generate_daily_recap_html(df, output_file="daily_evolution_recap.html"):
    import pandas as pd
    from datetime import datetime
    
    # Get unique dates sorted
    dates = df["Date"].sort_values().unique()
    
    # English day names (keeping them in English)
    english_days = {
        'Monday': 'Monday',
        'Tuesday': 'Tuesday', 
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday'
    }
    
    # English month names (keeping them in English)
    english_months = {
        'January': 'January',
        'February': 'February',
        'March': 'March',
        'April': 'April',
        'May': 'May',
        'June': 'June',
        'July': 'July',
        'August': 'August',
        'September': 'September',
        'October': 'October',
        'November': 'November',
        'December': 'December'
    }
    
    # Start HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Daily Evolution of H-1B Visa Narrative</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .date-section {
                background-color: white;
                margin-bottom: 30px;
                padding: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .date-header {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
                font-size: 1.5em;
                font-weight: 600;
            }
            .summary {
                color: #34495e;
                line-height: 1.6;
                margin-bottom: 20px;
            }
            .bullet-points {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .bullet-points h3 {
                color: #2c3e50;
                margin-top: 0;
                margin-bottom: 10px;
            }
            .bullet-points ul {
                margin: 0;
                padding-left: 20px;
            }
            .bullet-points li {
                margin-bottom: 8px;
                line-height: 1.4;
            }
            .quotes-section {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                border-left: 4px solid #3498db;
            }
            .quotes-section h3 {
                color: #2c3e50;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .quote-item {
                background-color: white;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 4px;
                border-left: 3px solid #95a5a6;
                font-style: italic;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <h1>Daily Recap: H-1B Visa Narrative - Donald Trump</h1>
    """
    
    # Add content for each date
    for date in dates:
        date_data = df[df["Date"] == date]
        
        # Format date in news style with English day and month names
        if isinstance(date, str):
            date_obj = pd.to_datetime(date)
        else:
            date_obj = date
            
        # Get English day and month names
        english_day = date_obj.strftime('%A')
        english_month = date_obj.strftime('%B')
        
        # Format: "Monday, 19 September 2025"
        formatted_date = f"{english_day}, {date_obj.day} {english_month} {date_obj.year}"
        
        # Handle None values safely
        summary = date_data["daily summary"].iloc[0]
        if summary is None or str(summary).strip() == "" or str(summary) == "nan":
            summary = "No summary available"
            
        bullet_points = date_data["daily bullet points"].iloc[0]
        
        # Get all quotes for this date
        quotes = date_data["Chunk Text"].dropna().unique()
        
        html_content += f"""
        <div class="date-section">
            <h2 class="date-header">{formatted_date}</h2>
            <div class="summary">
                <p>{summary}</p>
            </div>
            <div class="bullet-points">
                <h3>Key Points:</h3>
        """
        
        # Handle bullet points properly
        if bullet_points is None or str(bullet_points).strip() == "" or str(bullet_points) == "nan":
            html_content += "<p><em>No key points available</em></p>"
        else:
            # Convert bullet points to proper HTML list
            html_content += "<ul>"
            # Split by newlines and create list items
            points = str(bullet_points).split('\n')
            for point in points:
                point = point.strip()
                if point and point != "":
                    # Remove any existing bullet points (-, *, •)
                    point = point.lstrip('- *•').strip()
                    if point:
                        html_content += f"<li>{point}</li>"
            html_content += "</ul>"
        
        html_content += """
            </div>
            <div class="quotes-section">
        """
        html_content += f"<h3>Source Quotes ({len(quotes)} quotes):</h3>"
        
        # Add each quote
        for i, quote in enumerate(quotes, 1):
            # Handle None or empty quotes
            if quote and str(quote).strip() and str(quote) != "nan":
                html_content += f"""
                <div class="quote-item">
                    <strong>Quote {i}:</strong> {quote}
                </div>
                """
        
        html_content += """
            </div>
        </div>
        """
    
    # Close HTML structure
    html_content += """
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML file saved as: {output_file}")
    return output_file



def generate_html_report(result_df, original_df, output_file="company_report.html", quotes_limit=None):
    """
    Generate a formatted HTML report with summary, quotes, impact info and key points.
    
    Args:
        result_df: DataFrame with analysis results
        original_df: DataFrame with original quote data
        output_file: Name of the HTML output file
        quotes_limit: Maximum number of quotes per category (positive, negative, neutral).
                     If None, includes all quotes. If specified, limits to N quotes per category.
    """
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Company Analysis Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                overflow-x: hidden;
                box-sizing: border-box;
            }
            * {
                box-sizing: border-box;
            }
            .container * {
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow-x: hidden;
                box-sizing: border-box;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .company-card {
                margin: 30px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                max-width: 100%;
                width: 100%;
                box-sizing: border-box;
                display: block;
                position: relative;
            }
            .company-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
            }
            .company-name {
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }
            .company-meta {
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }
            .content-section {
                padding: 20px;
                overflow-x: hidden;
                word-wrap: break-word;
            }
            .section-title {
                color: #2c3e50;
                font-size: 18px;
                font-weight: bold;
                margin: 20px 0 10px 0;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }
            .summary {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin-bottom: 20px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
            }
            .quote {
                background-color: #fff3cd;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                font-style: italic;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
                box-sizing: border-box;
            }
            .quote-meta {
                font-size: 12px;
                color: #6c757d;
                margin-top: 10px;
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 3px;
                font-style: normal;
            }
            .impact-label {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                text-transform: uppercase;
                margin-right: 8px;
            }
            .impact-positive {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .impact-negative {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .impact-neutral {
                background-color: #e2e3e5;
                color: #383d41;
                border: 1px solid #d6d8db;
            }
            .key-points {
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #17a2b8;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
            }
            .key-point {
                margin: 8px 0;
                padding-left: 20px;
                position: relative;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
            }
            .key-point:before {
                content: "•";
                color: #17a2b8;
                font-weight: bold;
                position: absolute;
                left: 0;
            }
            .stats {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #f1f3f4;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                gap: 15px;
            }
            .stat-item {
                text-align: center;
                min-width: 60px;
                max-width: 100px;
            }
            .sentiment-group {
                display: flex;
                gap: 8px;
                align-items: center;
            }
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }
            .stat-label {
                font-size: 12px;
                color: #6c757d;
            }
            .meta-row {
                margin: 5px 0;
            }
            .meta-label {
                font-weight: bold;
                color: #495057;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Company Analysis Report</h1>
        </div>
    """
    
    for i, (company_name, row) in enumerate(result_df.iterrows(), 1):
        # Get company quotes
        company_data = original_df[original_df['Company'] == company_name]
        
        # Count impact labels
        impact_counts = company_data['Impact Label'].value_counts()
        positive_count = impact_counts.get('positive', 0)
        negative_count = impact_counts.get('negative', 0)
        neutral_count = impact_counts.get('neutral', 0)
        
        html_content += f"""
        <div class="container">
            <div class="company-card">
                <div class="company-header">
                    <h2 class="company-name">{company_name}</h2>
                    <div class="company-meta">
                        {row['sector']} | {row['industry']} | {row['ticker']} | {row['country']}
                    </div>
                </div>
                
                <div class="content-section">
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-number">{row['quote_count']}</div>
                            <div class="stat-label">Quotes</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{len(row['key_points'])}</div>
                            <div class="stat-label">Key Points</div>
                        </div>
                        <div class="sentiment-group">"""
        
        # Add impact counts only if greater than 0, grouped together
        if positive_count > 0:
            html_content += f"""
                            <div class="stat-item">
                                <div class="stat-number" style="color: #28a745;">{positive_count}</div>
                                <div class="stat-label">Positive</div>
                            </div>"""
        
        if negative_count > 0:
            html_content += f"""
                            <div class="stat-item">
                                <div class="stat-number" style="color: #dc3545;">{negative_count}</div>
                                <div class="stat-label">Negative</div>
                            </div>"""
        
        if neutral_count > 0:
            html_content += f"""
                            <div class="stat-item">
                                <div class="stat-number" style="color: #6c757d;">{neutral_count}</div>
                                <div class="stat-label">Neutral</div>
                            </div>"""
        
        html_content += """
                        </div>"""
        
        html_content += f"""
                    </div>
                    
                    <div class="section-title">📋 Summary</div>
                    <div class="summary">
                        {row['summary']}
                    </div>
                    
                    <div class="section-title">🎯 Key Points</div>
                    <div class="key-points">
        """
        
        if row['key_points'] and len(row['key_points']) > 0:
            for point in row['key_points']:
                html_content += f'\n                        <div class="key-point">{point}</div>'
        else:
            html_content += '\n                        <div class="key-point">No key points available</div>'
        
        html_content += f"""
                    </div>
                    
                    <div class="section-title">💬 Original Quotes</div>
        """
        
        # Filter and limit quotes by category if quotes_limit is specified
        if quotes_limit is not None and quotes_limit > 0:
            # Separate quotes by impact label
            positive_quotes = company_data[company_data['Impact Label'].str.lower() == 'positive'].head(quotes_limit)
            negative_quotes = company_data[company_data['Impact Label'].str.lower() == 'negative'].head(quotes_limit)
            neutral_quotes = company_data[company_data['Impact Label'].str.lower() == 'neutral'].head(quotes_limit)
            
            # Combine limited quotes, maintaining order: positive, negative, neutral
            import pandas as pd
            limited_quotes = pd.concat([positive_quotes, negative_quotes, neutral_quotes], ignore_index=True)
            quotes_to_show = limited_quotes
        else:
            # Show all quotes
            quotes_to_show = company_data
        
        for j, (_, quote_row) in enumerate(quotes_to_show.iterrows(), 1):
            # Determine impact label class
            impact_label = quote_row.get('Impact Label', '').lower()
            if impact_label == 'positive':
                impact_class = 'impact-positive'
            elif impact_label == 'negative':
                impact_class = 'impact-negative'
            else:
                impact_class = 'impact-neutral'
            
            html_content += f"""
                    <div class="quote">
                        "{quote_row['Quote']}"
                        <div class="quote-meta">
                            <div class="meta-row">
                                <span class="meta-label">Theme:</span> {quote_row['Theme']}
                            </div>
                            <div class="meta-row">
                                <span class="meta-label">Motivation:</span> {quote_row['Motivation']}
                            </div>
                            <div class="meta-row">
                                <span class="meta-label">Impact:</span> 
                                <span class="impact-label {impact_class}">{quote_row.get('Impact Label', 'N/A')}</span>
                                {quote_row.get('Impact Motivation', 'N/A')}
                            </div>
                        </div>
                    </div>
            """
        
        # Add info about quote limitation if applied
        if quotes_limit is not None and quotes_limit > 0:
            total_quotes = len(company_data)
            shown_quotes = len(quotes_to_show)
            if total_quotes > shown_quotes:
                html_content += f"""
                    <div style="text-align: center; margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-style: italic; color: #6c757d;">
                        Showing {shown_quotes} of {total_quotes} quotes (limited to {quotes_limit} per category)
                    </div>
                """
        
        html_content += """
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report saved as '{output_file}'")
    
    # Display in notebook
    from IPython.display import HTML, display
    display(HTML(html_content))
    
    return html_content


def generate_entity_evolution_html(df, output_file="entity_evolution_recap.html"):
    import pandas as pd
    from datetime import datetime
    
    # Get unique entities sorted
    entities = df["Entity"].sort_values().unique()
    
    # English day names and months (same as before)
    english_days = {
        'Monday': 'Monday', 'Tuesday': 'Tuesday', 'Wednesday': 'Wednesday',
        'Thursday': 'Thursday', 'Friday': 'Friday', 'Saturday': 'Saturday', 'Sunday': 'Sunday'
    }
    
    english_months = {
        'January': 'January', 'February': 'February', 'March': 'March',
        'April': 'April', 'May': 'May', 'June': 'June',
        'July': 'July', 'August': 'August', 'September': 'September',
        'October': 'October', 'November': 'November', 'December': 'December'
    }
    
    # Start HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Entity Evolution of H-1B Visa Narrative</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .entity-section {
                background-color: white;
                margin-bottom: 40px;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .entity-header {
                color: #e74c3c;
                border-bottom: 4px solid #e74c3c;
                padding-bottom: 15px;
                margin-bottom: 30px;
                font-size: 2em;
                font-weight: 700;
            }
            .date-subsection {
                background-color: #f8f9fa;
                margin-bottom: 25px;
                padding: 20px;
                border-radius: 6px;
                border-left: 4px solid #3498db;
            }
            .date-subheader {
                color: #2c3e50;
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 15px;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 8px;
            }
            .summary {
                color: #34495e;
                line-height: 1.6;
                margin-bottom: 20px;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
            }
            .bullet-points {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .bullet-points h4 {
                color: #2c3e50;
                margin-top: 0;
                margin-bottom: 10px;
            }
            .bullet-points ul {
                margin: 0;
                padding-left: 20px;
            }
            .bullet-points li {
                margin-bottom: 8px;
                line-height: 1.4;
            }
            .quotes-section {
                background-color: #fff;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
                border: 1px solid #ddd;
            }
            .quotes-section h4 {
                color: #2c3e50;
                margin-top: 0;
                margin-bottom: 15px;
            }
            .quote-item {
                background-color: #f8f9fa;
                padding: 12px;
                margin-bottom: 10px;
                border-radius: 4px;
                border-left: 3px solid #95a5a6;
                font-style: italic;
                line-height: 1.5;
                font-size: 0.95em;
            }
            .no-data {
                color: #7f8c8d;
                font-style: italic;
                text-align: center;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Entity Evolution Recap: H-1B Visa Narrative</h1>
    """
    
    # Add content for each entity
    for entity in entities:
        entity_data = df[df["Entity"] == entity]
        
        # Get unique dates for this entity, sorted
        entity_dates = entity_data["Date"].sort_values().unique()
        
        html_content += f"""
        <div class="entity-section">
            <h2 class="entity-header">{entity}</h2>
        """
        
        # Add content for each date for this entity
        for date in entity_dates:
            date_entity_data = entity_data[entity_data["Date"] == date]
            
            # Format date
            if isinstance(date, str):
                date_obj = pd.to_datetime(date)
            else:
                date_obj = date
                
            english_day = date_obj.strftime('%A')
            english_month = date_obj.strftime('%B')
            formatted_date = f"{english_day}, {date_obj.day} {english_month} {date_obj.year}"
            
            # Get enhanced summary and key points
            enhanced_summary = date_entity_data["enhanced_summary"].iloc[0] if len(date_entity_data) > 0 else None
            enhanced_key_points = date_entity_data["enhanced_key_points"].iloc[0] if len(date_entity_data) > 0 else None
            
            # Handle None values safely
            if enhanced_summary is None or str(enhanced_summary).strip() == "" or str(enhanced_summary) == "nan":
                enhanced_summary = "No enhanced summary available"
                
            # Get all quotes (Chunk Text) for this entity on this date
            quotes = date_entity_data["Chunk Text"].dropna().unique()
            
            html_content += f"""
            <div class="date-subsection">
                <h3 class="date-subheader">{formatted_date}</h3>
                <div class="summary">
                    <p>{enhanced_summary}</p>
                </div>
                <div class="bullet-points">
                    <h4>Enhanced Key Points:</h4>
            """
            
            # Handle enhanced key points
            if enhanced_key_points is None or str(enhanced_key_points).strip() == "" or str(enhanced_key_points) == "nan":
                html_content += "<p class='no-data'>No enhanced key points available</p>"
            else:
                html_content += "<ul>"
                # Split by newlines and create list items
                points = str(enhanced_key_points).split('\n')
                for point in points:
                    point = point.strip()
                    if point and point != "":
                        # Remove any existing bullet points (-, *, •)
                        point = point.lstrip('- *•').strip()
                        if point:
                            html_content += f"<li>{point}</li>"
                html_content += "</ul>"
            
            html_content += """
                </div>
                <div class="quotes-section">
            """
            html_content += f"<h4>Source References ({len(quotes)} quotes):</h4>"
            
            # Add each quote
            if len(quotes) > 0:
                for i, quote in enumerate(quotes, 1):
                    # Handle None or empty quotes
                    if quote and str(quote).strip() and str(quote) != "nan":
                        html_content += f"""
                        <div class="quote-item">
                            <strong>Reference {i}:</strong> {quote}
                        </div>
                        """
            else:
                html_content += "<p class='no-data'>No source references available</p>"
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Close HTML structure
    html_content += """
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Entity evolution HTML file saved as: {output_file}")
    return output_file

def generate_company_comparison_html(df, output_file="company_comparison_report.html"):
    import pandas as pd
    from datetime import datetime
    
    # Get unique entities sorted
    entities = df["Entity"].sort_values().unique()
    
    # English day names and months
    english_days = {
        'Monday': 'Monday', 'Tuesday': 'Tuesday', 'Wednesday': 'Wednesday',
        'Thursday': 'Thursday', 'Friday': 'Friday', 'Saturday': 'Saturday', 'Sunday': 'Sunday'
    }
    
    english_months = {
        'January': 'January', 'February': 'February', 'March': 'March',
        'April': 'April', 'May': 'May', 'June': 'June',
        'July': 'July', 'August': 'August', 'September': 'September',
        'October': 'October', 'November': 'November', 'December': 'December'
    }
    
    # Start HTML structure
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Company H-1B Visa Narrative Analysis Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }
            .company-section {
                background-color: white;
                margin-bottom: 50px;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 6px solid #e74c3c;
            }
            .company-header {
                color: #e74c3c;
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 30px;
                text-align: center;
                border-bottom: 3px solid #e74c3c;
                padding-bottom: 15px;
            }
            .date-subsection {
                background-color: #f8f9fa;
                margin-bottom: 35px;
                padding: 25px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                position: relative;
            }
            .date-subheader {
                color: #2c3e50;
                font-size: 1.4em;
                font-weight: 600;
                margin-bottom: 20px;
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 12px 20px;
                border-radius: 6px;
                margin: -25px -25px 20px -25px;
            }
            .enhanced-section {
                background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 25px;
                border-left: 4px solid #27ae60;
            }
            .enhanced-header {
                color: #27ae60;
                font-size: 1.2em;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            .enhanced-header::before {
                content: "✨";
                margin-right: 8px;
                font-size: 1.1em;
            }
            .enhanced-summary {
                color: #2c3e50;
                line-height: 1.7;
                margin-bottom: 15px;
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #27ae60;
            }
            .enhanced-key-points {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #27ae60;
            }
            .enhanced-key-points h4 {
                color: #27ae60;
                margin-top: 0;
                margin-bottom: 12px;
                font-weight: 600;
            }
            .enhanced-key-points ul {
                margin: 0;
                padding-left: 20px;
            }
            .enhanced-key-points li {
                margin-bottom: 8px;
                line-height: 1.5;
                color: #2c3e50;
            }
            .original-section {
                background: linear-gradient(135deg, #fdf2e9, #faf5f0);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #f39c12;
            }
            .original-header {
                color: #f39c12;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            .original-header::before {
                content: "📄";
                margin-right: 8px;
            }
            .original-summary {
                color: #2c3e50;
                line-height: 1.6;
                margin-bottom: 15px;
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #f39c12;
            }
            .key-points {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                border-left: 3px solid #f39c12;
            }
            .key-points h4 {
                color: #f39c12;
                margin-top: 0;
                margin-bottom: 12px;
                font-weight: 600;
            }
            .key-points ul {
                margin: 0;
                padding-left: 20px;
            }
            .key-points li {
                margin-bottom: 8px;
                line-height: 1.5;
                color: #2c3e50;
            }
            .quotes-section {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border-left: 3px solid #f39c12;
            }
            .quotes-section h4 {
                color: #f39c12;
                margin-top: 0;
                margin-bottom: 15px;
                font-weight: 600;
            }
            .quote-item {
                background-color: #fefefe;
                padding: 12px;
                margin-bottom: 10px;
                border-radius: 4px;
                border-left: 3px solid #bdc3c7;
                font-style: italic;
                line-height: 1.5;
                font-size: 0.95em;
                color: #555;
            }
            .no-data {
                color: #7f8c8d;
                font-style: italic;
                text-align: center;
                padding: 20px;
                background-color: #ecf0f1;
                border-radius: 6px;
            }
            .main-title {
                text-align: center;
                color: #2c3e50;
                font-size: 3em;
                font-weight: 700;
                margin-bottom: 20px;
                background: linear-gradient(135deg, #3498db, #e74c3c);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                font-size: 1.2em;
                margin-bottom: 40px;
            }
        </style>
    </head>
    <body>
        <h1 class="main-title">H-1B Visa Impact Analysis</h1>
        <p class="subtitle">Comprehensive Company-by-Company Narrative Evolution Report</p>
    """
    
    # Add content for each entity
    for entity in entities:
        entity_data = df[df["Entity"] == entity]
        
        # Get unique dates for this entity, sorted
        entity_dates = entity_data["Date"].sort_values().unique()
        
        html_content += f"""
        <div class="company-section">
            <h2 class="company-header">{entity}</h2>
        """
        
        # Add content for each date for this entity
        for date in entity_dates:
            date_entity_data = entity_data[entity_data["Date"] == date]
            
            # Format date
            if isinstance(date, str):
                date_obj = pd.to_datetime(date)
            else:
                date_obj = date
                
            english_day = date_obj.strftime('%A')
            english_month = date_obj.strftime('%B')
            formatted_date = f"{english_day}, {date_obj.day} {english_month} {date_obj.year}"
            
            # Get data
            enhanced_summary = date_entity_data["enhanced_summary"].iloc[0] if len(date_entity_data) > 0 else None
            enhanced_key_points = date_entity_data["enhanced_key_points"].iloc[0] if len(date_entity_data) > 0 else None
            original_summary = date_entity_data["Summary"].iloc[0] if len(date_entity_data) > 0 else None
            original_key_points = date_entity_data["Key_points"].iloc[0] if len(date_entity_data) > 0 else None
            quotes = date_entity_data["Quotes"].iloc[0] if len(date_entity_data) > 0 else None
            
            html_content += f"""
            <div class="date-subsection">
                <h3 class="date-subheader">{formatted_date}</h3>
                
                <!-- Enhanced Section -->
                <div class="enhanced-section">
                    <div class="enhanced-header">Enhanced Analysis</div>
                    <div class="enhanced-summary">
                        <strong>Enhanced Summary:</strong><br>
                        {enhanced_summary if enhanced_summary and str(enhanced_summary) != 'nan' else '<span class="no-data">No enhanced summary available</span>'}
                    </div>
                    <div class="enhanced-key-points">
                        <h4>Enhanced Key Points:</h4>
            """
            
            # Handle enhanced key points
            if enhanced_key_points and str(enhanced_key_points) != 'nan':
                if isinstance(enhanced_key_points, list):
                    if enhanced_key_points:
                        html_content += "<ul>"
                        for point in enhanced_key_points:
                            if point and str(point).strip():
                                html_content += f"<li>{point}</li>"
                        html_content += "</ul>"
                    else:
                        html_content += '<div class="no-data">No enhanced key points available</div>'
                else:
                    # If it's a string, split by newlines
                    points = str(enhanced_key_points).split('\n')
                    if any(point.strip() for point in points):
                        html_content += "<ul>"
                        for point in points:
                            point = point.strip()
                            if point:
                                point = point.lstrip('- *•').strip()
                                if point:
                                    html_content += f"<li>{point}</li>"
                        html_content += "</ul>"
                    else:
                        html_content += '<div class="no-data">No enhanced key points available</div>'
            else:
                html_content += '<div class="no-data">No enhanced key points available</div>'
            
            html_content += """
                    </div>
                </div>
                
                <!-- Original Section -->
                <div class="original-section">
                    <div class="original-header">Original Analysis</div>
                    <div class="original-summary">
                        <strong>Original Summary:</strong><br>
            """
            
            # Fix the f-string backslash issue
            no_data_message = '<span class="no-data">No original summary available</span>'
            summary_content = original_summary if original_summary and str(original_summary) != 'nan' else no_data_message
            html_content += summary_content
            
            html_content += """
                    </div>
                    <div class="key-points">
                        <h4>Original Key Points:</h4>
            """
            
            # Handle original key points
            if original_key_points and str(original_key_points) != 'nan':
                if isinstance(original_key_points, list):
                    if original_key_points:
                        html_content += "<ul>"
                        for point in original_key_points:
                            if point and str(point).strip():
                                html_content += f"<li>{point}</li>"
                        html_content += "</ul>"
                    else:
                        html_content += '<div class="no-data">No original key points available</div>'
                else:
                    points = str(original_key_points).split('\n')
                    if any(point.strip() for point in points):
                        html_content += "<ul>"
                        for point in points:
                            point = point.strip()
                            if point:
                                point = point.lstrip('- *•').strip()
                                if point:
                                    html_content += f"<li>{point}</li>"
                        html_content += "</ul>"
                    else:
                        html_content += '<div class="no-data">No original key points available</div>'
            else:
                html_content += '<div class="no-data">No original key points available</div>'
            
            html_content += """
                    </div>
                    <div class="quotes-section">
                        <h4>Source Quotes:</h4>
            """
            
            # Handle quotes
            if quotes and str(quotes) != 'nan':
                if isinstance(quotes, list):
                    if quotes:
                        for i, quote in enumerate(quotes, 1):
                            if quote and str(quote).strip():
                                html_content += f"""
                                <div class="quote-item">
                                    <strong>Quote {i}:</strong> {quote}
                                </div>
                                """
                    else:
                        html_content += '<div class="no-data">No quotes available</div>'
                else:
                    html_content += f"""
                    <div class="quote-item">
                        {quotes}
                    </div>
                    """
            else:
                html_content += '<div class="no-data">No quotes available</div>'
            
            html_content += """
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Close HTML structure
    html_content += """
    </body>
    </html>
    """
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Company comparison HTML file saved as: {output_file}")
    return output_file


def generate_companies_timeline_chart_html(df_highlights, output_file="companies_timeline_chart.html"):
    """
    Generate HTML report with timeline of company highlights grouped by region.
    
    Args:
        df_highlights: DataFrame with columns ['Date', 'Highlight', 'Companies']
        output_file: Output HTML file name
    """
    import pandas as pd
    from datetime import datetime
    
    # Clean up the date format (remove "Data " prefix)
    df = df_highlights.copy()
    # Convert Date column to datetime (dates should already be in YYYY-MM-DD format)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define company regions
    us_companies = ['Alphabet Inc.', 'Amazon.com Inc.', 'Microsoft Corp.']
    india_companies = ['Wipro Ltd.', 'Infosys Ltd.', 'Tata Consultancy Services Ltd.']
    
    # Classify highlights by region
    def get_region(companies_str):
        if any(company in companies_str for company in us_companies):
            return 'USA'
        elif any(company in companies_str for company in india_companies):
            return 'India'
        else:
            return 'Other'
    
    df['Region'] = df['Companies'].apply(get_region)
    
    # Get unique dates sorted
    dates = sorted(df['Date'].unique())
    
    # Group highlights by date and region
    timeline_data = {}
    for date in dates:
        timeline_data[date] = {
            'USA': df[(df['Date'] == date) & (df['Region'] == 'USA')]['Highlight'].tolist(),
            'India': df[(df['Date'] == date) & (df['Region'] == 'India')]['Highlight'].tolist()
        }
    
    # Generate timeline HTML
    timeline_html = ""
    for i, date in enumerate(dates):
        date_str = date.strftime('%B %d, %Y')
        day_name = date.strftime('%A')
        
        usa_highlights = timeline_data[date]['USA']
        india_highlights = timeline_data[date]['India']
        
        # Skip dates with no highlights
        if not usa_highlights and not india_highlights:
            continue
            
        timeline_html += f"""
        <div class="timeline-item">
            <div class="timeline-marker"></div>
            <div class="timeline-content">
                <div class="timeline-date">
                    <span class="day">{day_name}</span>
                    <span class="date">{date_str}</span>
                </div>
                
                <div class="regions-container">
        """
        
        # USA highlights
        if usa_highlights:
            timeline_html += f"""
                    <div class="region-block usa-block">
                        <h3 class="region-title">🇺🇸 USA Companies</h3>
                        <div class="companies-list">Alphabet Inc., Amazon.com Inc., Microsoft Corp.</div>
                        <div class="highlights">
            """
            for highlight in usa_highlights:
                timeline_html += f'<div class="highlight-item">• {highlight}</div>'
            timeline_html += """
                        </div>
                    </div>
            """
        
        # India highlights
        if india_highlights:
            timeline_html += f"""
                    <div class="region-block india-block">
                        <h3 class="region-title">🇮🇳 India Companies</h3>
                        <div class="companies-list">Wipro Ltd., Infosys Ltd., Tata Consultancy Services Ltd.</div>
                        <div class="highlights">
            """
            for highlight in india_highlights:
                timeline_html += f'<div class="highlight-item">• {highlight}</div>'
            timeline_html += """
                        </div>
                    </div>
            """
        
        timeline_html += """
                </div>
            </div>
        </div>
        """
    
    # Create complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Companies Timeline Report</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #2c3e50;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 50px;
                padding-bottom: 20px;
                border-bottom: 3px solid #3498db;
                font-size: 2.8em;
                font-weight: 300;
            }}
            .timeline {{
                position: relative;
                padding: 20px 0;
            }}
            .timeline::before {{
                content: '';
                position: absolute;
                left: 50%;
                top: 0;
                bottom: 0;
                width: 4px;
                background: linear-gradient(to bottom, #3498db, #2980b9);
                transform: translateX(-50%);
            }}
            .timeline-item {{
                position: relative;
                margin: 60px 0;
                display: flex;
                align-items: flex-start;
            }}
            .timeline-marker {{
                position: absolute;
                left: 50%;
                top: 20px;
                width: 16px;
                height: 16px;
                background: #3498db;
                border: 4px solid white;
                border-radius: 50%;
                transform: translateX(-50%);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                z-index: 10;
            }}
            .timeline-content {{
                width: 100%;
                display: flex;
                align-items: flex-start;
                gap: 40px;
            }}
            .timeline-date {{
                flex: 0 0 200px;
                text-align: right;
                padding-right: 40px;
                margin-top: 10px;
            }}
            .timeline-date .day {{
                display: block;
                font-size: 1.1em;
                font-weight: 600;
                color: #3498db;
            }}
            .timeline-date .date {{
                display: block;
                font-size: 0.95em;
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .regions-container {{
                flex: 1;
                margin-left: 40px;
            }}
            .region-block {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.08);
                border-left: 5px solid;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .region-block:hover {{
                transform: translateY(-3px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.12);
            }}
            .usa-block {{
                border-left-color: #e74c3c;
                background: linear-gradient(135deg, #fff 0%, #fff5f5 100%);
            }}
            .india-block {{
                border-left-color: #f39c12;
                background: linear-gradient(135deg, #fff 0%, #fffbf5 100%);
            }}
            .region-title {{
                font-size: 1.3em;
                font-weight: 600;
                margin: 0 0 10px 0;
                color: #2c3e50;
            }}
            .companies-list {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-bottom: 15px;
                font-style: italic;
            }}
            .highlights {{
                margin-top: 15px;
            }}
            .highlight-item {{
                background: rgba(52, 152, 219, 0.05);
                padding: 12px 15px;
                margin: 8px 0;
                border-radius: 8px;
                border-left: 3px solid #3498db;
                font-size: 0.95em;
                line-height: 1.6;
                transition: background 0.2s ease;
            }}
            .highlight-item:hover {{
                background: rgba(52, 152, 219, 0.1);
            }}
            .metadata {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }}
            .summary-stats {{
                display: flex;
                justify-content: center;
                gap: 40px;
                margin: 30px 0;
                flex-wrap: wrap;
            }}
            .stat-item {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
                display: block;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            @media (max-width: 768px) {{
                .timeline-content {{
                    flex-direction: column;
                    gap: 20px;
                }}
                .timeline-date {{
                    flex: none;
                    text-align: left;
                    padding-right: 0;
                    margin-bottom: 10px;
                }}
                .regions-container {{
                    margin-left: 0;
                }}
                .timeline::before {{
                    left: 20px;
                }}
                .timeline-marker {{
                    left: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🕒 Companies Timeline Report</h1>
            
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">{len(dates)}</span>
                    <div class="stat-label">Days Analyzed</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df)}</span>
                    <div class="stat-label">Total Highlights</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df[df['Region'] == 'USA'])}</span>
                    <div class="stat-label">USA Highlights</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df[df['Region'] == 'India'])}</span>
                    <div class="stat-label">India Highlights</div>
                </div>
            </div>
            
            <div class="timeline">
                {timeline_html}
            </div>
            
            <div class="metadata">
                <p>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                <p>Timeline covers {dates[0].strftime('%B %d')} to {dates[-1].strftime('%B %d, %Y')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Timeline report generated: {output_file}")
    return output_file


def generate_companies_line_chart_html(df_highlights, output_file="companies_line_chart.html"):
    """
    Generate HTML report with horizontal line chart showing company highlights with arrows.
    
    Args:
        df_highlights: DataFrame with columns ['Date', 'Highlight', 'Companies']
        output_file: Output HTML file name
    """
    import pandas as pd
    from datetime import datetime
    
    # Clean up the date format (remove "Data " prefix)
    df = df_highlights.copy()
    # Convert Date column to datetime (dates should already be in YYYY-MM-DD format)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define company regions
    us_companies = ['Alphabet Inc.', 'Amazon.com Inc.', 'Microsoft Corp.']
    india_companies = ['Wipro Ltd.', 'Infosys Ltd.', 'Tata Consultancy Services Ltd.']
    
    # Classify highlights by region
    def get_region(companies_str):
        if any(company in companies_str for company in us_companies):
            return 'USA'
        elif any(company in companies_str for company in india_companies):
            return 'India'
        else:
            return 'Other'
    
    df['Region'] = df['Companies'].apply(get_region)
    
    # Get unique dates sorted
    dates = sorted(df['Date'].unique())
    
    # Group highlights by date and region
    timeline_data = {}
    for date in dates:
        timeline_data[date] = {
            'USA': df[(df['Date'] == date) & (df['Region'] == 'USA')]['Highlight'].tolist(),
            'India': df[(df['Date'] == date) & (df['Region'] == 'India')]['Highlight'].tolist()
        }
    
    # Generate chart points HTML
    chart_points_html = ""
    total_days = len(dates)
    point_index = 0  # Counter for alternating positions
    
    for i, date in enumerate(dates):
        date_str = date.strftime('%Y-%m-%d')
        date_display = date.strftime('%b %d')
        
        usa_highlights = timeline_data[date]['USA']
        india_highlights = timeline_data[date]['India']
        
        # Skip dates with no highlights
        if not usa_highlights and not india_highlights:
            continue
        
        # Calculate position on the line (percentage)
        position = (i / (total_days - 1)) * 100 if total_days > 1 else 50
        
        # Determine if this point should be above or below the line (alternating)
        is_above = point_index % 2 == 0
        position_class = "above" if is_above else "below"
        
        chart_points_html += f"""
        <div class="chart-point {position_class}" style="left: {position}%;" data-date="{date_str}">
            <div class="point-marker"></div>
            <div class="date-label">{date_display}</div>
            <div class="highlights-box">
        """
        
        # Add US highlights section
        if usa_highlights:
            chart_points_html += """
                <div class="region-header">US Companies</div>
            """
            for highlight in usa_highlights:
                chart_points_html += f'<div class="highlight-item">• {highlight}</div>'
        
        # Add India highlights section
        if india_highlights:
            chart_points_html += """
                <div class="region-header">Indian Companies</div>
            """
            for highlight in india_highlights:
                chart_points_html += f'<div class="highlight-item">• {highlight}</div>'
        
        chart_points_html += """
                <div class="arrow"></div>
            </div>
        </div>
        """
        
        point_index += 1
    
    # Create complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Companies Timeline Chart</title>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #2c3e50;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 50px;
                font-size: 2.5em;
                font-weight: 300;
            }}
            .chart-container {{
                position: relative;
                margin: 300px 0;
                padding: 0 20px;
                min-height: 500px;
                max-width: 1000px;
                margin-left: auto;
                margin-right: auto;
            }}
            .timeline-line {{
                position: absolute;
                top: 50%;
                left: 20px;
                right: 20px;
                height: 4px;
                background: linear-gradient(to right, #3498db, #2980b9);
                border-radius: 2px;
                transform: translateY(-50%);
            }}
            .chart-point {{
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
            }}
            .point-marker {{
                width: 14px;
                height: 14px;
                background: #e74c3c;
                border: 3px solid white;
                border-radius: 50%;
                position: relative;
                transform: translateX(-50%);
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                z-index: 20;
            }}
            .date-label {{
                position: absolute;
                top: 35px;
                left: 0;
                transform: translateX(-50%);
                font-size: 0.75em;
                font-weight: 600;
                color: #3498db;
                white-space: nowrap;
                background: white;
                padding: 3px 6px;
                border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            .highlights-box {{
                position: absolute;
                left: 0;
                transform: translateX(-50%);
                background: white;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 15px;
                min-width: 280px;
                max-width: 320px;
                border: 2px solid #3498db;
                z-index: 10;
            }}
            /* Above positioning */
            .chart-point.above .highlights-box {{
                bottom: 60px;
            }}
            .chart-point.above .arrow {{
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-top: 10px solid #3498db;
            }}
            /* Below positioning */
            .chart-point.below .highlights-box {{
                top: 60px;
            }}
            .chart-point.below .arrow {{
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 10px solid #3498db;
            }}
            .region-header {{
                font-size: 0.9em;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0 8px 0;
                padding-bottom: 4px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .region-header:first-child {{
                margin-top: 0;
            }}
            .highlight-item {{
                background: rgba(52, 152, 219, 0.08);
                padding: 6px 10px;
                margin: 4px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
                font-size: 0.85em;
                line-height: 1.3;
            }}
            .summary-stats {{
                display: flex;
                justify-content: center;
                gap: 40px;
                margin: 30px 0;
                flex-wrap: wrap;
            }}
            .stat-item {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
                display: block;
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 5px;
            }}
            .instructions {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin: 30px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }}
            .metadata {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }}
            @media (max-width: 768px) {{
                .chart-container {{
                    padding: 0 20px;
                }}
                .highlights-popup {{
                    min-width: 300px;
                    max-width: 350px;
                }}
                .summary-stats {{
                    gap: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Companies Timeline Chart</h1>
            
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">{len(dates)}</span>
                    <div class="stat-label">Days Analyzed</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df)}</span>
                    <div class="stat-label">Total Highlights</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df[df['Region'] == 'USA'])}</span>
                    <div class="stat-label">USA Highlights</div>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(df[df['Region'] == 'India'])}</span>
                    <div class="stat-label">India Highlights</div>
                </div>
            </div>
            
            <div class="instructions">
                📍 Red markers show key dates with highlights displayed alternately above and below the timeline
            </div>
            
            <div class="chart-container">
                <div class="timeline-line"></div>
                {chart_points_html}
            </div>
            
            <div class="metadata">
                <p>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                <p>Timeline covers {dates[0].strftime('%B %d')} to {dates[-1].strftime('%B %d, %Y')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Line chart report generated: {output_file}")
    return output_file


def generate_combined_dashboard_html(df_highlights, plotly_fig, output_file="combined_dashboard.html"):
    """
    Generate combined HTML report with both timeline chart and Plotly interactive dashboard.
    
    Args:
        df_highlights: DataFrame with highlights
        plotly_fig: Plotly figure object from create_enhanced_interactive_chart
        output_file: Output HTML file name
    """
    import pandas as pd
    from datetime import datetime
    
    # Generate timeline chart HTML directly (don't use temp file)
    # Clean up the date format
    df_timeline = df_highlights.copy()
    # Convert Date column to datetime (dates should already be in YYYY-MM-DD format)
    df_timeline['Date'] = pd.to_datetime(df_timeline['Date'])
    
    # Define company regions
    us_companies = ['Alphabet Inc.', 'Amazon.com Inc.', 'Microsoft Corp.']
    india_companies = ['Wipro Ltd.', 'Infosys Ltd.', 'Tata Consultancy Services Ltd.']
    
    # Classify highlights by region
    def get_region(companies_str):
        if any(company in companies_str for company in us_companies):
            return 'USA'
        elif any(company in companies_str for company in india_companies):
            return 'India'
        else:
            return 'Other'
    
    df_timeline['Region'] = df_timeline['Companies'].apply(get_region)
    
    # Get unique dates sorted
    timeline_dates = sorted(df_timeline['Date'].unique())
    
    # Group highlights by date and region
    timeline_data = {}
    for date in timeline_dates:
        timeline_data[date] = {
            'USA': df_timeline[(df_timeline['Date'] == date) & (df_timeline['Region'] == 'USA')]['Highlight'].tolist(),
            'India': df_timeline[(df_timeline['Date'] == date) & (df_timeline['Region'] == 'India')]['Highlight'].tolist()
        }
    
    # Generate chart points HTML
    chart_points_html = ""
    total_days = len(timeline_dates)
    point_index = 0
    
    for i, date in enumerate(timeline_dates):
        date_str = date.strftime('%Y-%m-%d')
        date_display = date.strftime('%b %d')
        
        usa_highlights = timeline_data[date]['USA']
        india_highlights = timeline_data[date]['India']
        
        # Skip dates with no highlights
        if not usa_highlights and not india_highlights:
            continue
        
        # Calculate position on the line (percentage)
        position = (i / (total_days - 1)) * 100 if total_days > 1 else 50
        
        # Determine if this point should be above or below the line (alternating)
        is_above = point_index % 2 == 0
        position_class = "above" if is_above else "below"
        
        chart_points_html += f"""
        <div class="chart-point {position_class}" style="left: {position}%;" data-date="{date_str}">
            <div class="point-marker"></div>
            <div class="date-label">{date_display}</div>
            <div class="highlights-box">
        """
        
        # Add US highlights section
        if usa_highlights:
            chart_points_html += """
                <div class="region-header">US Companies</div>
            """
            for highlight in usa_highlights:
                chart_points_html += f'<div class="highlight-item">• {highlight}</div>'
        
        # Add India highlights section
        if india_highlights:
            chart_points_html += """
                <div class="region-header">Indian Companies</div>
            """
            for highlight in india_highlights:
                chart_points_html += f'<div class="highlight-item">• {highlight}</div>'
        
        chart_points_html += """
                <div class="arrow"></div>
            </div>
        </div>
        """
        
        point_index += 1
    
    # Create the complete chart HTML
    chart_html = f"""
        <div class="chart-container">
            <div class="timeline-line"></div>
            {chart_points_html}
        </div>
    """
    
    # Get Plotly HTML as div
    plotly_html = plotly_fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
    
    # Extract just the plotly div
    import re
    plotly_match = re.search(r'<div id="plotly-chart".*?</script>', plotly_html, re.DOTALL)
    plotly_div = plotly_match.group(0) if plotly_match else ""
    
    # Use the dates from timeline processing
    dates = timeline_dates
    
    # Create combined HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>H-1B Visa Fee Impact - Complete Dashboard</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #2c3e50;
                min-height: 100vh;
            }}
            .main-container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header-section {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header-section h1 {{
                margin: 0;
                font-size: 2.8em;
                font-weight: 300;
            }}
            .header-section p {{
                margin: 15px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content-section {{
                padding: 60px 40px;
            }}
            .section-title {{
                font-size: 2em;
                color: #2c3e50;
                margin: 60px 0 30px 0;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 15px;
            }}
            .section-description {{
                text-align: center;
                color: #7f8c8d;
                font-size: 1.1em;
                margin-bottom: 30px;
                line-height: 1.6;
            }}
            .dashboard-container {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 30px 0;
                border: 1px solid #ecf0f1;
            }}
            /* Timeline styles - copied from existing function */
            .chart-container {{
                position: relative;
                margin: 350px 0 200px 0;
                padding: 0 20px;
                min-height: 600px;
                max-width: 1000px;
                margin-left: auto;
                margin-right: auto;
            }}
            .timeline-line {{
                position: absolute;
                top: 50%;
                left: 20px;
                right: 20px;
                height: 4px;
                background: linear-gradient(to right, #3498db, #2980b9);
                border-radius: 2px;
                transform: translateY(-50%);
            }}
            .chart-point {{
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
            }}
            .point-marker {{
                width: 14px;
                height: 14px;
                background: #e74c3c;
                border: 3px solid white;
                border-radius: 50%;
                position: relative;
                transform: translateX(-50%);
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                z-index: 20;
            }}
            .date-label {{
                position: absolute;
                top: 35px;
                left: 0;
                transform: translateX(-50%);
                font-size: 0.75em;
                font-weight: 600;
                color: #3498db;
                white-space: nowrap;
                background: white;
                padding: 3px 6px;
                border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            .highlights-box {{
                position: absolute;
                left: 0;
                transform: translateX(-50%);
                background: white;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 15px;
                min-width: 280px;
                max-width: 320px;
                border: 2px solid #3498db;
                z-index: 10;
            }}
            .chart-point.above .highlights-box {{
                bottom: 60px;
            }}
            .chart-point.above .arrow {{
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-top: 10px solid #3498db;
            }}
            .chart-point.below .highlights-box {{
                top: 60px;
            }}
            .chart-point.below .arrow {{
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 10px solid #3498db;
            }}
            .region-header {{
                font-size: 0.9em;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0 8px 0;
                padding-bottom: 4px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .region-header:first-child {{
                margin-top: 0;
            }}
            .highlight-item {{
                background: rgba(52, 152, 219, 0.08);
                padding: 6px 10px;
                margin: 4px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
                font-size: 0.85em;
                line-height: 1.3;
            }}
            .metadata {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="header-section">
                <h1>📊 H-1B Visa Fee Impact Dashboard</h1>
                <p>Comprehensive analysis of the $100,000 H-1B visa fee impact on companies</p>
            </div>
            
            <div class="content-section">
                <div class="section-title">📈 Interactive Data Analysis</div>
                <div class="section-description">
                    Explore the data with interactive charts showing narrative trends, company citations, and market analysis
                </div>
                <div class="dashboard-container">
                    {plotly_div}
                </div>
                
                <div class="section-title">⏱️ Timeline of Key Highlights</div>
                <div class="section-description">
                    Daily highlights showing the most important developments for US and Indian companies
                </div>
                {chart_html}
                
                <div class="metadata">
                    <p>Combined dashboard generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p>Timeline covers {dates[0].strftime('%B %d')} to {dates[-1].strftime('%B %d, %Y')}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write combined HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Combined dashboard generated: {output_file}")
    return output_file


def generate_interactive_timeline_dashboard_html(df_highlights, df_company_summaries, plotly_fig, output_file="interactive_dashboard.html"):
    """
    Generate HTML report with interactive timeline that can switch between country mode and company mode.
    
    Args:
        df_highlights: DataFrame with highlights by countries
        df_company_summaries: DataFrame with Entity and enhanced_summary columns  
        plotly_fig: Plotly figure object
        output_file: Output HTML file name
    """
    import pandas as pd
    from datetime import datetime
    import json
    
    # Generate timeline data for countries mode (existing logic)
    df_timeline = df_highlights.copy()
    # Convert Date column to datetime (dates should already be in YYYY-MM-DD format)
    df_timeline['Date'] = pd.to_datetime(df_timeline['Date'])
    
    us_companies = ['Alphabet Inc.', 'Amazon.com Inc.', 'Microsoft Corp.']
    india_companies = ['Wipro Ltd.', 'Infosys Ltd.', 'Tata Consultancy Services Ltd.']
    
    def get_region(companies_str):
        if any(company in companies_str for company in us_companies):
            return 'USA'
        elif any(company in companies_str for company in india_companies):
            return 'India'
        else:
            return 'Other'
    
    df_timeline['Region'] = df_timeline['Companies'].apply(get_region)
    timeline_dates = sorted(df_timeline['Date'].unique())
    
    # Generate countries timeline data
    countries_timeline_data = {}
    for date in timeline_dates:
        countries_timeline_data[date.strftime('%Y-%m-%d')] = {
            'USA': df_timeline[(df_timeline['Date'] == date) & (df_timeline['Region'] == 'USA')]['Highlight'].tolist(),
            'India': df_timeline[(df_timeline['Date'] == date) & (df_timeline['Region'] == 'India')]['Highlight'].tolist()
        }
    
    # Generate company summaries data using enhanced_summary field
    company_summaries_data = {}
    if 'Entity' in df_company_summaries.columns and 'enhanced_summary' in df_company_summaries.columns:
        
        # Check if df_company_summaries has date-specific data
        has_date_column = 'Date' in df_company_summaries.columns
        
        for entity in df_company_summaries['Entity'].unique():
            entity_data = df_company_summaries[df_company_summaries['Entity'] == entity]
            
            company_date_content = {}
            
            if has_date_column:
                # Use date-specific enhanced_summary
                for _, row in entity_data.iterrows():
                    row_date = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                    if row_date in [d.strftime('%Y-%m-%d') for d in timeline_dates]:
                        company_date_content[row_date] = row['enhanced_summary']
            else:
                # Use same enhanced_summary for all timeline dates where company appears
                enhanced_summary = entity_data.iloc[0]['enhanced_summary']
                
                # Find dates where this company appears in highlights 
                for date in timeline_dates:
                    date_str = date.strftime('%Y-%m-%d')
                    date_highlights = df_timeline[df_timeline['Date'] == date]
                    
                    # Check if company appears in any highlight for this date
                    company_appears = any(entity in companies for companies in date_highlights['Companies'])
                    
                    if company_appears:
                        company_date_content[date_str] = enhanced_summary
            
            if company_date_content:  # Only include companies that have data
                company_summaries_data[entity] = {
                    'date_content': company_date_content,
                    'general_summary': entity_data.iloc[0]['enhanced_summary']
                }
    
    # Get Plotly HTML
    plotly_html = plotly_fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
    import re
    plotly_match = re.search(r'<div id="plotly-chart".*?</script>', plotly_html, re.DOTALL)
    plotly_div = plotly_match.group(0) if plotly_match else ""
    
    # Get list of available companies
    available_companies = list(company_summaries_data.keys())
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive H-1B Visa Timeline Dashboard</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #2c3e50;
                min-height: 100vh;
            }}
            .main-container {{
                max-width: 1600px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .header-section {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header-section h1 {{
                margin: 0;
                font-size: 2.8em;
                font-weight: 300;
            }}
            .header-section p {{
                margin: 15px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .content-section {{
                padding: 60px 40px;
            }}
            .section-title {{
                font-size: 2em;
                color: #2c3e50;
                margin: 60px 0 30px 0;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 15px;
            }}
            .section-description {{
                text-align: center;
                color: #7f8c8d;
                font-size: 1.1em;
                margin-bottom: 30px;
                line-height: 1.6;
            }}
            .dashboard-container {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 30px 0;
                border: 1px solid #ecf0f1;
            }}
            .timeline-controls {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #ecf0f1;
            }}
            .control-group {{
                display: inline-block;
                margin: 0 20px;
                vertical-align: top;
            }}
            .control-label {{
                display: block;
                font-weight: bold;
                margin-bottom: 8px;
                color: #2c3e50;
            }}
            select {{
                padding: 8px 12px;
                border: 2px solid #3498db;
                border-radius: 6px;
                font-size: 14px;
                background: white;
                color: #2c3e50;
                min-width: 150px;
            }}
            select:focus {{
                outline: none;
                border-color: #2980b9;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
            }}
            .chart-container {{
                position: relative;
                margin: 350px 0 200px 0;
                padding: 0 20px;
                min-height: 600px;
                max-width: 1000px;
                margin-left: auto;
                margin-right: auto;
            }}
            .timeline-line {{
                position: absolute;
                top: 50%;
                left: 20px;
                right: 20px;
                height: 4px;
                background: linear-gradient(to right, #3498db, #2980b9);
                border-radius: 2px;
                transform: translateY(-50%);
            }}
            .chart-point {{
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
            }}
            .point-marker {{
                width: 14px;
                height: 14px;
                background: #e74c3c;
                border: 3px solid white;
                border-radius: 50%;
                position: relative;
                transform: translateX(-50%);
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
                z-index: 20;
            }}
            .date-label {{
                position: absolute;
                top: 35px;
                left: 0;
                transform: translateX(-50%);
                font-size: 0.75em;
                font-weight: 600;
                color: #3498db;
                white-space: nowrap;
                background: white;
                padding: 3px 6px;
                border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            .highlights-box {{
                position: absolute;
                left: 0;
                transform: translateX(-50%);
                background: white;
                border-radius: 10px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                padding: 15px;
                min-width: 280px;
                max-width: 320px;
                border: 2px solid #3498db;
                z-index: 10;
            }}
            .chart-point.above .highlights-box {{
                bottom: 60px;
            }}
            .chart-point.above .arrow {{
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-top: 10px solid #3498db;
            }}
            .chart-point.below .highlights-box {{
                top: 60px;
            }}
            .chart-point.below .arrow {{
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 10px solid #3498db;
            }}
            .region-header {{
                font-size: 0.9em;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0 8px 0;
                padding-bottom: 4px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .region-header:first-child {{
                margin-top: 0;
            }}
            .highlight-item {{
                background: rgba(52, 152, 219, 0.08);
                padding: 6px 10px;
                margin: 4px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
                font-size: 0.85em;
                line-height: 1.3;
            }}
            .company-summary {{
                background: rgba(52, 152, 219, 0.08);
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                font-size: 0.9em;
                line-height: 1.4;
                text-align: justify;
            }}
            .metadata {{
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="header-section">
                <h1>📊 Interactive H-1B Visa Timeline Dashboard</h1>
                <p>Switch between country highlights and individual company summaries</p>
            </div>
            
            <div class="content-section">
                <div class="section-title">📈 Interactive Data Analysis</div>
                <div class="section-description">
                    Explore the data with interactive charts showing narrative trends, company citations, and market analysis
                </div>
                <div class="dashboard-container">
                    {plotly_div}
                </div>
                
                <div class="section-title">⏱️ Interactive Timeline</div>
                <div class="section-description">
                    Choose between country highlights or individual company summaries
                </div>
                
                <div class="timeline-controls">
                    <div class="control-group">
                        <label class="control-label">Timeline Mode:</label>
                        <select id="modeSelect" onchange="changeTimelineMode()">
                            <option value="countries">Countries Mode</option>
                            <option value="company">Company Mode</option>
                        </select>
                    </div>
                    <div class="control-group" id="companyGroup" style="display: none;">
                        <label class="control-label">Select Company:</label>
                        <select id="companySelect" onchange="changeCompany()">
                            {chr(10).join(f'<option value="{company}">{company}</option>' for company in available_companies)}
                        </select>
                    </div>
                </div>
                
                <div class="chart-container">
                    <div class="timeline-line"></div>
                    <div id="timelineContent">
                        <!-- Timeline content will be populated by JavaScript -->
                    </div>
                </div>
                
                <div class="metadata">
                    <p>Interactive dashboard generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                    <p>Timeline covers {timeline_dates[0].strftime('%B %d')} to {timeline_dates[-1].strftime('%B %d, %Y')}</p>
                </div>
            </div>
        </div>
        
        <script>
            // Timeline data
            const countriesData = {json.dumps(countries_timeline_data)};
            const companiesData = {json.dumps(company_summaries_data)};
            
            let currentMode = 'countries';
            let currentCompany = '{available_companies[0] if available_companies else ""}';
            
            function changeTimelineMode() {{
                const mode = document.getElementById('modeSelect').value;
                const companyGroup = document.getElementById('companyGroup');
                
                currentMode = mode;
                
                if (mode === 'company') {{
                    companyGroup.style.display = 'inline-block';
                    updateTimelineCompany();
                }} else {{
                    companyGroup.style.display = 'none';
                    updateTimelineCountries();
                }}
            }}
            
            function changeCompany() {{
                currentCompany = document.getElementById('companySelect').value;
                updateTimelineCompany();
            }}
            
            function updateTimelineCountries() {{
                const container = document.getElementById('timelineContent');
                let html = '';
                let pointIndex = 0;
                
                const sortedDates = Object.keys(countriesData).sort();
                const totalDays = sortedDates.length;
                
                sortedDates.forEach((dateStr, index) => {{
                    const data = countriesData[dateStr];
                    const usaHighlights = data.USA || [];
                    const indiaHighlights = data.India || [];
                    
                    if (usaHighlights.length === 0 && indiaHighlights.length === 0) return;
                    
                    const position = totalDays > 1 ? (index / (totalDays - 1)) * 100 : 50;
                    const isAbove = pointIndex % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    
                    html += `
                        <div class="chart-point ${{positionClass}}" style="left: ${{position}}%; top: 50%;">
                            <div class="point-marker"></div>
                            <div class="date-label">${{dateDisplay}}</div>
                            <div class="highlights-box">`;
                    
                    if (usaHighlights.length > 0) {{
                        html += '<div class="region-header">US Companies</div>';
                        usaHighlights.forEach(highlight => {{
                            html += `<div class="highlight-item">• ${{highlight}}</div>`;
                        }});
                    }}
                    
                    if (indiaHighlights.length > 0) {{
                        html += '<div class="region-header">Indian Companies</div>';
                        indiaHighlights.forEach(highlight => {{
                            html += `<div class="highlight-item">• ${{highlight}}</div>`;
                        }});
                    }}
                    
                    html += `
                                <div class="arrow"></div>
                            </div>
                        </div>`;
                    
                    pointIndex++;
                }});
                
                container.innerHTML = html;
            }}
            
            function updateTimelineCompany() {{
                const container = document.getElementById('timelineContent');
                const companyData = companiesData[currentCompany];
                
                if (!companyData) {{
                    container.innerHTML = '<p style="text-align: center; color: #7f8c8d;">No data available for selected company</p>';
                    return;
                }}
                
                let html = '';
                const dates = Object.keys(companyData.date_content).sort();
                const totalDays = dates.length;
                
                dates.forEach((dateStr, index) => {{
                    const position = totalDays > 1 ? (index / (totalDays - 1)) * 100 : 50;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    const dateContent = companyData.date_content[dateStr];
                    
                    html += `
                        <div class="chart-point ${{positionClass}}" style="left: ${{position}}%; top: 50%;">
                            <div class="point-marker"></div>
                            <div class="date-label">${{dateDisplay}}</div>
                            <div class="highlights-box">
                                <div class="region-header">${{currentCompany}}</div>
                                <div class="company-summary">${{dateContent}}</div>
                                <div class="arrow"></div>
                            </div>
                        </div>`;
                }});
                
                container.innerHTML = html;
            }}
            
            // Initialize with countries mode
            updateTimelineCountries();
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive timeline dashboard generated: {output_file}")
    return output_file
