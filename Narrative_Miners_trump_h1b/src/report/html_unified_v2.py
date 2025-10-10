"""
Dashboard HTML generation using unified DataFrame
This version copies the EXACT timeline design from the original html.py
"""

def _generate_summary_section(final_summary):
    """Helper function to generate the summary section HTML."""
    import json
    
    if isinstance(final_summary, dict):
        summary_text = final_summary.get("summary", "No general summary available.")
    elif isinstance(final_summary, str):
        if final_summary.strip().startswith('{"summary":'):
            try:
                parsed_dict = json.loads(final_summary)
                summary_text = parsed_dict.get("summary", "No general summary available.")
            except json.JSONDecodeError:
                summary_text = final_summary
        else:
            summary_text = final_summary
    else:
        summary_text = str(final_summary)
    
    summary_html = summary_text.replace("\n", "<br>").replace("\\n", "<br>")
    
    return f'''
        <div class="summary-section">
            <div class="summary-title">📰 Narrative Overview</div>
            <div class="summary-content">
                {summary_html}
            </div>
        </div>
    '''


def generate_entities_reports_html_unified(df_entities_data, countries_dict, people_entities_reports, df_entity_stats=None, unique_sentences_count=None):
    """Generate HTML for entity reports with statistics."""
    import pandas as pd
    
    html_content = """
        <div class="content-section">
            <div class="section-title">Entity Reports</div>
            <div class="entities-container">
    """
    
    # === PEOPLE SECTION FIRST ===
    if len(people_entities_reports) > 0:
        html_content += '<div class="region-section"><div class="region-title">👥 People</div><div class="entities-grid">'
        
        for _, row in people_entities_reports.iterrows():
            entity_name = row['Entity']
            summary = row['Final Summary'] if pd.notna(row['Final Summary']) else "No summary available"
            
            # Get statistics for this person
            stats_html = ''
            if df_entity_stats is not None and entity_name in df_entity_stats['Entity'].values:
                entity_stats = df_entity_stats[df_entity_stats['Entity'] == entity_name].iloc[0]
                overall_sentences = int(entity_stats['Overall_Total_Sentences']) if pd.notna(entity_stats.get('Overall_Total_Sentences')) else 0
                overall_documents = int(entity_stats['Overall_Unique_Documents']) if pd.notna(entity_stats.get('Overall_Unique_Documents')) else 0
                overall_percentage = float(entity_stats['Overall_Percentage_Documents']) if pd.notna(entity_stats.get('Overall_Percentage_Documents')) else 0
                
                # Calculate sentence percentage
                sentence_percentage = round((overall_sentences / unique_sentences_count * 100), 2) if unique_sentences_count and unique_sentences_count > 0 else 0
                
                stats_html = f'''
                    <div class="entity-stats">
                        <div class="stat-item">
                            <span class="stat-label">News Relevance:</span> 
                            <span class="stat-value">{overall_sentences} / {sentence_percentage}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">News Coverage:</span> 
                            <span class="stat-value">{overall_documents} / {overall_percentage:.1f}%</span>
                        </div>
                    </div>
                '''
            
            html_content += f'''
            <div class="entity-card">
                <div class="entity-header">
                    <h3 class="entity-name">{entity_name}</h3>
                    {stats_html}
                </div>
                <div class="entity-summary"><p>{summary}</p></div>
            </div>
            '''
        
        html_content += '</div></div>'
    
    # === COUNTRIES SECTIONS ===
    for country in sorted(countries_dict.keys()):
        country_entities = df_entities_data[df_entities_data['Region'] == country]
        
        if len(country_entities) > 0:
            html_content += f'<div class="region-section"><div class="region-title">🌍 {country}</div><div class="entities-grid">'
            
            for _, row in country_entities.iterrows():
                entity_name = row['Entity']
                summary = row['Final Summary'] if pd.notna(row['Final Summary']) else "No summary available"
                
                # Get statistics for this entity
                stats_html = ''
                if df_entity_stats is not None and entity_name in df_entity_stats['Entity'].values:
                    entity_stats = df_entity_stats[df_entity_stats['Entity'] == entity_name].iloc[0]
                    overall_sentences = int(entity_stats['Overall_Total_Sentences']) if pd.notna(entity_stats.get('Overall_Total_Sentences')) else 0
                    overall_documents = int(entity_stats['Overall_Unique_Documents']) if pd.notna(entity_stats.get('Overall_Unique_Documents')) else 0
                    overall_percentage = float(entity_stats['Overall_Percentage_Documents']) if pd.notna(entity_stats.get('Overall_Percentage_Documents')) else 0
                    
                    # Calculate sentence percentage
                    sentence_percentage = round((overall_sentences / unique_sentences_count * 100), 2) if unique_sentences_count and unique_sentences_count > 0 else 0
                    
                    stats_html = f'''
                        <div class="entity-stats">
                            <div class="stat-item">
                                <span class="stat-label">News Relevance:</span> 
                                <span class="stat-value">{overall_sentences} / {sentence_percentage}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">News Coverage:</span> 
                                <span class="stat-value">{overall_documents} / {overall_percentage:.1f}%</span>
                            </div>
                        </div>
                    '''
                
                html_content += f'''
                <div class="entity-card">
                    <div class="entity-header">
                        <h3 class="entity-name">{entity_name}</h3>
                        {stats_html}
                    </div>
                    <div class="entity-summary"><p>{summary}</p></div>
                </div>
                '''
            
            html_content += '</div></div>'
    
    html_content += '</div></div>'
    
    return html_content


def generate_interactive_timeline_dashboard_from_unified_df(
    df_final_news_coverage, 
    plotly_fig, 
    final_summary=None,
    df_entity_stats=None,
    unique_sentences_count=None,
    output_file="interactive_dashboard.html"
):
    """
    Generate HTML dashboard with ORIGINAL timeline design + navigation for >7 days.
    Takes unified DataFrame as input.
    
    Args:
        df_entity_stats: Optional DataFrame with entity statistics
        unique_sentences_count: Total number of unique sentences for percentage calculation
    """
    import pandas as pd
    import json
    import re
    
    df = df_final_news_coverage.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    timeline_dates = sorted(df['Date'].unique())
    
    # Generate countries_dict
    df_non_people = df[df['Country Code'].notna()].copy()
    countries_dict = {}
    for country in df_non_people['Country Code'].unique():
        if pd.notna(country):
            entities = df_non_people[df_non_people['Country Code'] == country]['Entity'].unique().tolist()
            countries_dict[country] = entities
    
    available_countries = list(countries_dict.keys())
    default_country = available_countries[0] if available_countries else 'Unknown'
    
    # Generate countries timeline data - with BOTH highlights and summaries (with entity names)
    countries_timeline_data = {}
    for date in timeline_dates:
        date_str = date.strftime('%Y-%m-%d')
        date_data = {}
        for country in available_countries:
            country_entities = countries_dict[country]
            country_df = df[
                (df['Date'] == date) & 
                (df['Entity'].isin(country_entities))
            ]
            
            # Get highlights with entity names - group by entity to avoid duplicates
            highlights_with_entities = []
            highlights_by_entity = {}
            
            for _, row in country_df[country_df['Highlights'].notna()].iterrows():
                entity_name = row['Entity']
                highlights_raw = row['Highlights']
                
                if entity_name not in highlights_by_entity:
                    highlights_by_entity[entity_name] = set()
                
                if isinstance(highlights_raw, list):
                    for h in highlights_raw:
                        highlights_by_entity[entity_name].add(h)
                elif isinstance(highlights_raw, str):
                    if highlights_raw.startswith('[') and highlights_raw.endswith(']'):
                        try:
                            import ast
                            parsed = ast.literal_eval(highlights_raw)
                            if isinstance(parsed, list):
                                for h in parsed:
                                    highlights_by_entity[entity_name].add(h)
                            else:
                                highlights_by_entity[entity_name].add(highlights_raw)
                        except:
                            highlights_by_entity[entity_name].add(highlights_raw)
                    else:
                        highlights_by_entity[entity_name].add(highlights_raw)
            
            # Convert to list format
            for entity_name, highlights_set in highlights_by_entity.items():
                for h in highlights_set:
                    highlights_with_entities.append({'entity': entity_name, 'text': h})
            
            # Get summaries with entity names - use set to ensure uniqueness
            summaries_with_entities = []
            summaries_by_entity = {}
            
            for _, row in country_df[country_df['Novel Daily Summary'].notna()].iterrows():
                entity_name = row['Entity']
                summary = row['Novel Daily Summary']
                
                # Store as tuple (entity, summary) to track unique combinations
                if entity_name not in summaries_by_entity:
                    summaries_by_entity[entity_name] = set()
                
                summaries_by_entity[entity_name].add(summary)
            
            # Convert to list format - should only have one summary per entity
            for entity_name, summaries_set in summaries_by_entity.items():
                for summary in summaries_set:
                    summaries_with_entities.append({'entity': entity_name, 'text': summary})
            
            date_data[country] = {
                'highlights': highlights_with_entities,
                'summaries': summaries_with_entities
            }
        countries_timeline_data[date_str] = date_data
    
    # Generate company summaries data - with BOTH summaries and highlights
    df_companies = df[(df['Entity Type'].isin(['COMP', 'ORGA'])) & (df['Country Code'].notna())].copy()
    company_summaries_data = {}
    
    for entity in df_companies['Entity'].unique():
        entity_data = df_companies[df_companies['Entity'] == entity]
        company_date_content = {}
        company_date_highlights = {}
        
        for date in timeline_dates:
            date_str = date.strftime('%Y-%m-%d')
            date_rows = entity_data[entity_data['Date'] == date]
            
            # Get summary
            if len(date_rows) > 0 and 'Novel Daily Summary' in date_rows.columns:
                summary = date_rows['Novel Daily Summary'].iloc[0]
                company_date_content[date_str] = summary if pd.notna(summary) else "No summary available"
            else:
                company_date_content[date_str] = "No summary available"
            
            # Get highlights
            if len(date_rows) > 0 and 'Highlights' in date_rows.columns:
                highlights_raw = date_rows['Highlights'].iloc[0]
                highlights_parsed = []
                
                if pd.notna(highlights_raw):
                    if isinstance(highlights_raw, list):
                        highlights_parsed = highlights_raw
                    elif isinstance(highlights_raw, str):
                        if highlights_raw.startswith('[') and highlights_raw.endswith(']'):
                            try:
                                import ast
                                parsed = ast.literal_eval(highlights_raw)
                                if isinstance(parsed, list):
                                    highlights_parsed = parsed
                                else:
                                    highlights_parsed = [highlights_raw]
                            except:
                                highlights_parsed = [highlights_raw]
                        else:
                            highlights_parsed = [highlights_raw]
                
                company_date_highlights[date_str] = highlights_parsed
            else:
                company_date_highlights[date_str] = []
        
        final_summary_entity = entity_data['Final Summary'].iloc[0] if 'Final Summary' in entity_data.columns and len(entity_data) > 0 else ""
        company_summaries_data[entity] = {
            'date_content': company_date_content,
            'date_highlights': company_date_highlights,
            'general_summary': final_summary_entity if pd.notna(final_summary_entity) else "No final summary"
        }
    
    available_companies = list(company_summaries_data.keys())
    
    # Generate people summaries data
    df_people = df[df['Entity Type'] == 'PEOP'].copy()
    people_summaries_data = {}
    available_people = []
    
    if len(df_people) > 0:
        available_people = sorted(df_people['Entity'].unique())
        
        for person in available_people:
            person_data = df_people[df_people['Entity'] == person]
            person_dates = sorted(person_data['Date'].unique())
            
            people_summaries_data[person] = {
                'dates': [d.strftime('%Y-%m-%d') for d in person_dates],
                'date_content': {}
            }
            
            for date in person_dates:
                date_str = date.strftime('%Y-%m-%d')
                day_data = person_data[person_data['Date'] == date]
                
                if len(day_data) > 0:
                    row = day_data.iloc[0]
                    summary = row.get('Novel Daily Summary', '') if pd.notna(row.get('Novel Daily Summary')) else ''
                    highlights_raw = row.get('Highlights', '') if pd.notna(row.get('Highlights')) else ''
                    
                    # Parse highlights if they are a list
                    highlights_str = ''
                    if isinstance(highlights_raw, list):
                        highlights_str = '\n'.join(highlights_raw)
                    elif isinstance(highlights_raw, str):
                        if highlights_raw.startswith('[') and highlights_raw.endswith(']'):
                            try:
                                import ast
                                parsed = ast.literal_eval(highlights_raw)
                                if isinstance(parsed, list):
                                    highlights_str = '\n'.join(parsed)
                                else:
                                    highlights_str = highlights_raw
                            except:
                                highlights_str = highlights_raw
                        else:
                            highlights_str = highlights_raw
                    
                    people_summaries_data[person]['date_content'][date_str] = {
                        'summary': summary,
                        'key_points': highlights_str
                    }
    
    # Generate entity reports
    df_entities_data = df.groupby('Entity').agg({
        'Final Summary': 'first',
        'Entity Type': 'first',
        'Country Code': 'first'
    }).reset_index()
    
    df_entities_data = df_entities_data[df_entities_data['Final Summary'].notna()]
    
    def get_entity_region(entity_name):
        for country, entity_list in countries_dict.items():
            if entity_name in entity_list:
                return country
        return 'Other'
    
    df_entities_data['Region'] = df_entities_data['Entity'].apply(get_entity_region)
    
    all_companies_from_countries = []
    for company_list in countries_dict.values():
        all_companies_from_countries.extend(company_list)
    all_companies_from_countries = set(all_companies_from_countries)
    
    other_entities = df_entities_data[df_entities_data['Region'] == 'Other']
    people_entities_reports = other_entities[~other_entities['Entity'].isin(all_companies_from_countries)]
    
    entities_html = generate_entities_reports_html_unified(df_entities_data, countries_dict, people_entities_reports, df_entity_stats, unique_sentences_count)
    
    # Get Plotly HTML
    plotly_html = plotly_fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
    plotly_match = re.search(r'<div id="plotly-chart".*?</script>', plotly_html, re.DOTALL)
    plotly_div = plotly_match.group(0) if plotly_match else ""
    
    summary_section_html = _generate_summary_section(final_summary) if final_summary else ""
    
    # Country checkboxes HTML
    country_checkboxes = ''.join([
        f'<label class="checkbox-label"><input type="checkbox" id="country_{c}" value="{c}" onchange="updateSelectedCountries()" checked> {c}</label>'
        for c in available_countries
    ])
    
    # Company checkboxes HTML
    company_checkboxes = ''.join([
        f'<label class="checkbox-label"><input type="checkbox" id="company_{c}" value="{c}" onchange="updateSelectedCompanies()" checked> {c}</label>'
        for c in available_companies
    ])
    
    # People checkboxes HTML - only first 3 checked by default
    people_checkboxes = ''.join([
        f'<label class="checkbox-label"><input type="checkbox" id="person_{p}" value="{p}" onchange="updateSelectedPeople()" {"checked" if i < 3 else ""}> {p}</label>'
        for i, p in enumerate(available_people)
    ])
    
    # Build HTML with ORIGINAL timeline design
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Timeline Dashboard</title>
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
                max-width: 1700px;
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
            .timeline-controls {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
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
                min-width: 150px;
            }}
            .navigation-controls {{
                text-align: center;
                margin: 20px 0;
            }}
            .nav-button {{
                padding: 10px 20px;
                margin: 0 10px;
                font-size: 16px;
                background: #3498db;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
            }}
            .nav-button:disabled {{
                background: #bdc3c7;
                cursor: not-allowed;
            }}
            .date-range-display {{
                display: inline-block;
                margin: 0 20px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .chart-container {{
                position: relative;
                margin: 350px 0 200px 0;
                padding: 0 30px;
                min-height: 600px;
                max-width: 1200px;
                margin-left: auto;
                margin-right: auto;
            }}
            .timeline-line {{
                position: absolute;
                top: 50%;
                left: 30px;
                right: 30px;
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
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
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
                min-width: 360px;
                max-width: 400px;
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
                z-index: 15;
            }}
            .chart-point.below .highlights-box {{
                top: 85px;
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
                z-index: 15;
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
                line-height: 0.91;
            }}
            .company-summary {{
                background: rgba(52, 152, 219, 0.08);
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                font-size: 0.9em;
                line-height: 0.98;
                text-align: justify;
            }}
            .checkbox-container {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin: 15px 0;
                max-height: 300px;
                overflow-y: auto;
                padding: 10px;
                background: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }}
            .checkbox-label {{
                display: flex;
                align-items: center;
                cursor: pointer;
                padding: 8px 12px;
                border-radius: 5px;
                transition: background 0.2s;
                white-space: nowrap;
            }}
            .checkbox-label:hover {{
                background: rgba(52, 152, 219, 0.1);
            }}
            .checkbox-label input {{
                margin-right: 10px;
                cursor: pointer;
            }}
            .entities-container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            .region-section {{
                margin-bottom: 50px;
            }}
            .region-title {{
                color: #2c3e50;
                font-size: 1.8em;
                margin-bottom: 30px;
                text-align: center;
                font-weight: 600;
            }}
            .entities-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 25px;
                margin-bottom: 40px;
            }}
            .entity-card {{
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border: 1px solid #e9ecef;
            }}
            .entity-header {{
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 15px;
                margin-bottom: 20px;
            }}
            .entity-name {{
                color: #2c3e50;
                font-size: 1.3em;
                font-weight: 600;
                margin: 0;
            }}
            .entity-summary {{
                color: #34495e;
                line-height: 1.6;
            }}
            .summary-section {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 30px;
                margin: 30px 0;
                border-left: 5px solid #3498db;
            }}
            .summary-title {{
                font-size: 1.8em;
                color: #2c3e50;
                margin-bottom: 20px;
                font-weight: 600;
            }}
            .summary-content {{
                color: #34495e;
                line-height: 1.8;
                text-align: justify;
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="header-section">
                <h1>📊 Interactive Timeline Dashboard</h1>
                <p>Explore narrative developments across time, entities, and regions</p>
            </div>
            
            <div class="content-section">
                {summary_section_html}
                
                <div class="section-title">⏱️ Interactive Timeline</div>
                
                <div class="timeline-controls">
                    <div class="control-group">
                        <label class="control-label">View Mode:</label>
                        <select id="modeSelect" onchange="changeTimelineMode()">
                            <option value="countries">Countries</option>
                            <option value="company">Companies</option>
                            <option value="people" selected>People</option>
                        </select>
                    </div>
                    
                    <div class="control-group" id="countryGroup" style="display:none;">
                        <label class="control-label">Select Countries:</label>
                        <div class="checkbox-container">
                            {country_checkboxes}
                        </div>
                    </div>
                    
                    <div class="control-group" id="companyGroup" style="display:none;">
                        <label class="control-label">Select Companies:</label>
                        <div class="checkbox-container">
                            {company_checkboxes}
                        </div>
                    </div>
                    
                    <div class="control-group" id="peopleGroup">
                        <label class="control-label">Select People:</label>
                        <div class="checkbox-container">
                            {people_checkboxes}
                        </div>
                    </div>
                    
                    <div class="control-group" id="contentGroup">
                        <label class="control-label">Content Type:</label>
                        <select id="contentTypeSelect" onchange="renderCurrentMode()">
                            <option value="summary" selected>Novel Daily Summary</option>
                            <option value="highlights">Highlights</option>
                        </select>
                    </div>
                </div>
                
                <div class="navigation-controls">
                    <button class="nav-button" id="prevBtn" onclick="navigatePrev()">← Previous 7 Days</button>
                    <span class="date-range-display" id="dateRange"></span>
                    <button class="nav-button" id="nextBtn" onclick="navigateNext()">Next 7 Days →</button>
                </div>
                
                <div class="chart-container">
                    <div class="timeline-line"></div>
                    <div id="timelineContent"></div>
                </div>
                
                <div class="section-title">📈 Interactive Charts</div>
                {plotly_div}
            </div>
            
            {entities_html}
        </div>
        
        <script>
            const countriesData = {json.dumps(countries_timeline_data)};
            const companiesData = {json.dumps(company_summaries_data)};
            const availableCountries = {json.dumps(available_countries)};
            const availableCompanies = {json.dumps(available_companies)};
            const peopleData = {json.dumps(people_summaries_data)};
            const availablePeople = {json.dumps(available_people)};
            
            let currentMode = 'people';
            let selectedCountries = availableCountries.slice();
            let selectedCompanies = availableCompanies.slice();
            let selectedPeople = availablePeople.slice(0, 3);  // Select only first 3
            let allDates = [];
            let currentStartIndex = 0;
            const DAYS_TO_SHOW = 7;
            
            function updateAllDates() {{
                if (currentMode === 'countries') {{
                    allDates = Object.keys(countriesData).sort();
                }} else if (currentMode === 'company') {{
                    // Get all dates from selected companies
                    const datesSet = new Set();
                    selectedCompanies.forEach(company => {{
                        const companyData = companiesData[company];
                        if (companyData) {{
                            Object.keys(companyData.date_content).forEach(date => datesSet.add(date));
                        }}
                    }});
                    allDates = Array.from(datesSet).sort();
                }} else if (currentMode === 'people') {{
                    // Get all dates from selected people
                    const datesSet = new Set();
                    selectedPeople.forEach(person => {{
                        const personData = peopleData[person];
                        if (personData) {{
                            personData.dates.forEach(date => datesSet.add(date));
                        }}
                    }});
                    allDates = Array.from(datesSet).sort();
                }}
            }}
            
            function navigatePrev() {{
                if (currentStartIndex > 0) {{
                    currentStartIndex = Math.max(0, currentStartIndex - DAYS_TO_SHOW);
                    renderCurrentMode();
                }}
            }}
            
            function navigateNext() {{
                if (currentStartIndex + DAYS_TO_SHOW < allDates.length) {{
                    currentStartIndex = Math.min(allDates.length - DAYS_TO_SHOW, currentStartIndex + DAYS_TO_SHOW);
                    renderCurrentMode();
                }}
            }}
            
            function updateNavButtons() {{
                document.getElementById('prevBtn').disabled = (currentStartIndex === 0);
                document.getElementById('nextBtn').disabled = (currentStartIndex + DAYS_TO_SHOW >= allDates.length);
                
                if (allDates.length > 0) {{
                    const visibleDates = allDates.slice(currentStartIndex, currentStartIndex + DAYS_TO_SHOW);
                    const start = new Date(visibleDates[0]).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    const end = new Date(visibleDates[visibleDates.length - 1]).toLocaleDateString('en-US', {{month: 'short', day: 'numeric', year: 'numeric'}});
                    document.getElementById('dateRange').textContent = `${{start}} - ${{end}}`;
                }} else {{
                    document.getElementById('dateRange').textContent = 'No dates';
                }}
            }}
            
            function renderCurrentMode() {{
                if (currentMode === 'countries') {{
                    updateTimelineCountries();
                }} else if (currentMode === 'company') {{
                    updateTimelineCompany();
                }} else {{
                    updateTimelinePeople();
                }}
                updateNavButtons();
            }}
            
            function changeTimelineMode() {{
                const mode = document.getElementById('modeSelect').value;
                currentMode = mode;
                currentStartIndex = 0;
                
                document.getElementById('companyGroup').style.display = mode === 'company' ? 'inline-block' : 'none';
                document.getElementById('countryGroup').style.display = mode === 'countries' ? 'inline-block' : 'none';
                document.getElementById('peopleGroup').style.display = mode === 'people' ? 'inline-block' : 'none';
                
                updateAllDates();
                renderCurrentMode();
            }}
            
            function updateSelectedCountries() {{
                selectedCountries = [];
                availableCountries.forEach(country => {{
                    const checkbox = document.getElementById(`country_${{country}}`);
                    if (checkbox && checkbox.checked) {{
                        selectedCountries.push(country);
                    }}
                }});
                if (selectedCountries.length === 0) {{
                    selectedCountries = [availableCountries[0]];
                    document.getElementById(`country_${{availableCountries[0]}}`).checked = true;
                }}
                renderCurrentMode();
            }}
            
            function updateSelectedCompanies() {{
                selectedCompanies = [];
                availableCompanies.forEach(company => {{
                    const checkbox = document.getElementById(`company_${{company}}`);
                    if (checkbox && checkbox.checked) {{
                        selectedCompanies.push(company);
                    }}
                }});
                if (selectedCompanies.length === 0) {{
                    selectedCompanies = [availableCompanies[0]];
                    document.getElementById(`company_${{availableCompanies[0]}}`).checked = true;
                }}
                updateAllDates();
                renderCurrentMode();
            }}
            
            function updateSelectedPeople() {{
                selectedPeople = [];
                availablePeople.forEach(person => {{
                    const checkbox = document.getElementById(`person_${{person}}`);
                    if (checkbox && checkbox.checked) {{
                        selectedPeople.push(person);
                    }}
                }});
                if (selectedPeople.length === 0) {{
                    selectedPeople = [availablePeople[0]];
                    document.getElementById(`person_${{availablePeople[0]}}`).checked = true;
                }}
                updateAllDates();
                renderCurrentMode();
            }}
            
            function updateTimelineCountries() {{
                const container = document.getElementById('timelineContent');
                const visibleDates = allDates.slice(currentStartIndex, currentStartIndex + DAYS_TO_SHOW);
                const contentType = document.getElementById('contentTypeSelect').value;
                
                // Collect all dates with their content (or lack thereof)
                const dateItems = [];
                visibleDates.forEach(dateStr => {{
                    const data = countriesData[dateStr];
                    let dateHasContent = false;
                    let dateContentHtml = '';
                    
                    selectedCountries.forEach(country => {{
                        const countryData = data[country] || {{}};
                        let countryContentHtml = '';
                        
                        if (contentType === 'highlights') {{
                            const highlights = countryData.highlights || [];
                            if (highlights.length > 0) {{
                                dateHasContent = true;
                                countryContentHtml += `<div class="region-header">${{country}} Companies</div>`;
                                const highlightsByEntity = {{}};
                                highlights.forEach(item => {{
                                    if (!highlightsByEntity[item.entity]) {{
                                        highlightsByEntity[item.entity] = [];
                                    }}
                                    highlightsByEntity[item.entity].push(item.text);
                                }});
                                Object.keys(highlightsByEntity).forEach(entityName => {{
                                    countryContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{entityName}}</strong></div>`;
                                    highlightsByEntity[entityName].forEach(text => {{
                                        countryContentHtml += `<div class="highlight-item">• ${{text}}</div>`;
                                    }});
                                }});
                            }}
                        }} else {{
                            const summaries = countryData.summaries || [];
                            if (summaries.length > 0) {{
                                dateHasContent = true;
                                countryContentHtml += `<div class="region-header">${{country}} Companies</div>`;
                                summaries.forEach(item => {{
                                    countryContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{item.entity}}</strong></div>`;
                                    countryContentHtml += `<div class="company-summary">${{item.text}}</div>`;
                                }});
                            }}
                        }}
                        
                        dateContentHtml += countryContentHtml;
                    }});
                    
                    dateItems.push({{dateStr: dateStr, hasContent: dateHasContent, content: dateContentHtml}});
                }});
                
                // Render all dates (with or without content)
                let html = '';
                const totalDays = dateItems.length;
                dateItems.forEach((item, index) => {{
                    const position = totalDays > 1 ? (index / (totalDays - 1)) * 100 : 50;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(item.dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    
                    if (item.hasContent) {{
                        // Show point with content box
                        html += `<div class="chart-point ${{positionClass}}" style="left: ${{position}}%;"><div class="point-marker"></div><div class="date-label">${{dateDisplay}}</div><div class="highlights-box">`;
                        html += item.content;
                        html += '<div class="arrow"></div></div></div>';
                    }} else {{
                        // Show empty point (no content box)
                        html += `<div class="chart-point" style="left: ${{position}}%;"><div class="point-marker" style="background: #bdc3c7;"></div><div class="date-label">${{dateDisplay}}</div></div>`;
                    }}
                }});
                
                container.innerHTML = html;
            }}
            
            function updateTimelineCompany() {{
                const container = document.getElementById('timelineContent');
                const contentType = document.getElementById('contentTypeSelect').value;
                
                // Special case: single entity with < 7 dates, show all
                let visibleDates;
                if (selectedCompanies.length === 1 && allDates.length < DAYS_TO_SHOW) {{
                    visibleDates = allDates;
                }} else {{
                    visibleDates = allDates.slice(currentStartIndex, currentStartIndex + DAYS_TO_SHOW);
                }}
                
                // Collect all dates with their content
                const dateItems = [];
                visibleDates.forEach(dateStr => {{
                    let dateHasContent = false;
                    let dateContentHtml = '';
                    
                    selectedCompanies.forEach(company => {{
                        const companyData = companiesData[company];
                        if (!companyData) return;
                        
                        let companyContentHtml = '';
                        
                        if (contentType === 'summary') {{
                            const dateContent = companyData.date_content[dateStr];
                            if (dateContent && dateContent !== 'No summary available') {{
                                dateHasContent = true;
                                companyContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{company}}</strong></div>`;
                                companyContentHtml += `<div class="company-summary">${{dateContent}}</div>`;
                            }}
                        }} else {{
                            const highlights = companyData.date_highlights[dateStr] || [];
                            if (highlights.length > 0) {{
                                dateHasContent = true;
                                companyContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{company}}</strong></div>`;
                                highlights.forEach(highlight => {{
                                    companyContentHtml += `<div class="highlight-item">• ${{highlight}}</div>`;
                                }});
                            }}
                        }}
                        
                        dateContentHtml += companyContentHtml;
                    }});
                    
                    dateItems.push({{dateStr: dateStr, hasContent: dateHasContent, content: dateContentHtml}});
                }});
                
                // Render all dates
                let html = '';
                const totalDays = dateItems.length;
                dateItems.forEach((item, index) => {{
                    const position = totalDays > 1 ? (index / (totalDays - 1)) * 100 : 50;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(item.dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    
                    if (item.hasContent) {{
                        html += `<div class="chart-point ${{positionClass}}" style="left: ${{position}}%;"><div class="point-marker"></div><div class="date-label">${{dateDisplay}}</div><div class="highlights-box">`;
                        html += item.content;
                        html += '<div class="arrow"></div></div></div>';
                    }} else {{
                        html += `<div class="chart-point" style="left: ${{position}}%;"><div class="point-marker" style="background: #bdc3c7;"></div><div class="date-label">${{dateDisplay}}</div></div>`;
                    }}
                }});
                
                container.innerHTML = html;
            }}
            
            function updateTimelinePeople() {{
                const container = document.getElementById('timelineContent');
                const contentType = document.getElementById('contentTypeSelect').value;
                
                // Special case: single entity with < 7 dates, show all
                let visibleDates;
                if (selectedPeople.length === 1 && allDates.length < DAYS_TO_SHOW) {{
                    visibleDates = allDates;
                }} else {{
                    visibleDates = allDates.slice(currentStartIndex, currentStartIndex + DAYS_TO_SHOW);
                }}
                
                // Collect all dates with their content
                const dateItems = [];
                visibleDates.forEach(dateStr => {{
                    let dateHasContent = false;
                    let dateContentHtml = '';
                    
                    selectedPeople.forEach(person => {{
                        const personData = peopleData[person];
                        if (!personData) return;
                        
                        const dateContent = personData.date_content[dateStr];
                        if (!dateContent) return;
                        
                        let personContentHtml = '';
                        
                        if (contentType === 'summary') {{
                            const content = dateContent.summary || '';
                            if (content && content !== 'No summary available') {{
                                dateHasContent = true;
                                personContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{person}}</strong></div>`;
                                personContentHtml += `<div class="company-summary">${{content}}</div>`;
                            }}
                        }} else {{
                            const highlights = dateContent.key_points || '';
                            if (highlights) {{
                                const points = highlights.split('\\n').filter(point => point.trim());
                                if (points.length > 0) {{
                                    dateHasContent = true;
                                    personContentHtml += `<div style="margin-top: 12px;"><strong style="color: #2c3e50; font-size: 0.95em;">${{person}}</strong></div>`;
                                    points.forEach(point => {{
                                        personContentHtml += `<div class="highlight-item">• ${{point.trim()}}</div>`;
                                    }});
                                }}
                            }}
                        }}
                        
                        dateContentHtml += personContentHtml;
                    }});
                    
                    dateItems.push({{dateStr: dateStr, hasContent: dateHasContent, content: dateContentHtml}});
                }});
                
                // Render all dates
                let html = '';
                const totalDays = dateItems.length;
                dateItems.forEach((item, index) => {{
                    const position = totalDays > 1 ? (index / (totalDays - 1)) * 100 : 50;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(item.dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    
                    if (item.hasContent) {{
                        html += `<div class="chart-point ${{positionClass}}" style="left: ${{position}}%;"><div class="point-marker"></div><div class="date-label">${{dateDisplay}}</div><div class="highlights-box">`;
                        html += item.content;
                        html += '<div class="arrow"></div></div></div>';
                    }} else {{
                        html += `<div class="chart-point" style="left: ${{position}}%;"><div class="point-marker" style="background: #bdc3c7;"></div><div class="date-label">${{dateDisplay}}</div></div>`;
                    }}
                }});
                
                container.innerHTML = html;
            }}
            
            window.onload = function() {{
                changeTimelineMode();
            }};
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive timeline dashboard generated: {output_file}")
    return output_file
