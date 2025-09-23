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