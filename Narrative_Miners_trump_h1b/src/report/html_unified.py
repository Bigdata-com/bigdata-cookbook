
def _generate_summary_section(final_summary):
    """Helper function to generate the summary section HTML with proper newline handling."""
    import json
    
    # Handle different input types
    if isinstance(final_summary, dict):
        summary_text = final_summary.get("summary", "No general summary available.")
    elif isinstance(final_summary, str):
        # Check if it's a JSON string that looks like a dict
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
    
    # Replace newlines with HTML breaks
    summary_html = summary_text.replace("\n", "<br>").replace("\\n", "<br>")
    
    return f'''
            <div class="content-section">
                <div class="section-title">📰 Narrative Overview</div>
                <div class="summary-box">
                    <p>{summary_html}</p>
                </div>
            </div>
            '''


def generate_entities_reports_html_unified(df_entities_data, countries_dict, people_entities_reports):
    """
    Generate HTML section for individual entity reports from unified DataFrame.
    
    Args:
        df_entities_data: DataFrame with Entity, Final Summary, Entity Type, Region
        countries_dict: Dictionary with country names as keys and lists of entities as values
        people_entities_reports: DataFrame with people entities filtered
    
    Returns:
        str: HTML content for entities reports section
    """
    import pandas as pd
    
    html_content = """
        <div class="content-section" id="entitiesSection">
            <h2 class="section-title">📋 Entity Reports</h2>
            <div class="section-description">
                Detailed summaries for each entity organized by region and type
            </div>
    """
    
    # === Companies and Organizations by Country ===
    for country in sorted(countries_dict.keys()):
        country_entities = df_entities_data[df_entities_data['Region'] == country]
        
        if len(country_entities) > 0:
            html_content += f"""
            <div class="region-section">
                <div class="region-title">🌍 {country}</div>
            """
            
            for _, row in country_entities.iterrows():
                entity_name = row['Entity']
                summary = row['Final Summary'] if pd.notna(row['Final Summary']) else "No summary available"
                
                html_content += f"""
                <div class="entity-card">
                    <h3>{entity_name}</h3>
                    <p>{summary}</p>
                </div>
                """
            
            html_content += """
            </div>
            """
    
    # === People Section ===
    if len(people_entities_reports) > 0:
        html_content += """
        <div class="region-section">
            <div class="region-title">👥 People</div>
        """
        
        for _, row in people_entities_reports.iterrows():
            entity_name = row['Entity']
            summary = row['Final Summary'] if pd.notna(row['Final Summary']) else "No summary available"
            
            html_content += f"""
            <div class="entity-card">
                <h3>{entity_name}</h3>
                <p>{summary}</p>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    html_content += """
        </div>
    """
    
    return html_content


def generate_interactive_timeline_dashboard_from_unified_df(
    df_final_news_coverage, 
    plotly_fig, 
    final_summary=None, 
    output_file="interactive_dashboard.html"
):
    """
    Generate HTML report with interactive timeline using a unified DataFrame.
    Uses the original timeline visualization (chart-points on line) with navigation for >7 days.
    
    Args:
        df_final_news_coverage: Unified DataFrame with columns:
            - Entity: str (entity name)
            - Entity Type: str ('PEOP', 'COMP', 'ORGA')
            - Country Code: str or None (None for people)
            - Date: datetime
            - Highlights: str (daily highlights for entity)
            - Novel Daily Summary: str (daily summary for entity)
            - Final Summary: str (final summary across all days)
        plotly_fig: Plotly figure object
        final_summary: Optional string or dict with general narrative summary
        output_file: Output HTML file name
    """
    import pandas as pd
    from datetime import datetime
    import json
    
    # Ensure Date is datetime
    df = df_final_news_coverage.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get unique dates sorted
    timeline_dates = sorted(df['Date'].unique())
    
    # === STEP 1: Generate countries_dict from Country Code ===
    df_non_people = df[df['Country Code'].notna()].copy()
    
    countries_dict = {}
    for country in df_non_people['Country Code'].unique():
        if pd.notna(country):
            entities = df_non_people[df_non_people['Country Code'] == country]['Entity'].unique().tolist()
            countries_dict[country] = entities
    
    available_countries = list(countries_dict.keys())
    
    # === STEP 2: Generate highlights timeline data (Countries Mode) ===
    countries_timeline_data = {}
    
    for date in timeline_dates:
        date_str = date.strftime('%Y-%m-%d')
        date_data = {}
        
        for country in available_countries:
            country_entities = countries_dict[country]
            highlights = df[
                (df['Date'] == date) & 
                (df['Entity'].isin(country_entities)) &
                (df['Highlights'].notna())
            ]['Highlights'].unique().tolist()
            
            date_data[country] = highlights
        
        countries_timeline_data[date_str] = date_data
    
    # === STEP 3: Generate company summaries data (Companies Mode) ===
    df_companies = df[
        (df['Entity Type'].isin(['COMP', 'ORGA'])) & 
        (df['Country Code'].notna())
    ].copy()
    
    company_summaries_data = {}
    
    for entity in df_companies['Entity'].unique():
        entity_data = df_companies[df_companies['Entity'] == entity]
        company_date_content = {}
        
        for date in timeline_dates:
            date_str = date.strftime('%Y-%m-%d')
            date_rows = entity_data[entity_data['Date'] == date]
            
            if len(date_rows) > 0 and 'Novel Daily Summary' in date_rows.columns:
                summary = date_rows['Novel Daily Summary'].iloc[0]
                company_date_content[date_str] = summary if pd.notna(summary) else "No summary available for this date"
            else:
                company_date_content[date_str] = "No summary available for this date"
        
        final_summary_entity = entity_data['Final Summary'].iloc[0] if 'Final Summary' in entity_data.columns and len(entity_data) > 0 else ""
        
        company_summaries_data[entity] = {
            'date_content': company_date_content,
            'general_summary': final_summary_entity if pd.notna(final_summary_entity) else "No final summary available"
        }
    
    available_companies = list(company_summaries_data.keys())
    
    # === STEP 4: Generate people summaries data (People Mode) ===
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
                    highlights = row.get('Highlights', '') if pd.notna(row.get('Highlights')) else ''
                    
                    content_parts = []
                    if summary:
                        content_parts.append(f"<strong>Summary:</strong><br>{summary}")
                    if highlights:
                        content_parts.append(f"<strong>Highlights:</strong><br>{highlights}")
                    
                    people_summaries_data[person]['date_content'][date_str] = {
                        'summary': summary,
                        'highlights': highlights
                    }
    
    # === STEP 5: Generate entity reports data ===
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
    
    entities_html = generate_entities_reports_html_unified(df_entities_data, countries_dict, people_entities_reports)
    
    # === STEP 6: Get Plotly HTML ===
    plotly_html = plotly_fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
    import re
    plotly_match = re.search(r'<div id="plotly-chart".*?</script>', plotly_html, re.DOTALL)
    plotly_div = plotly_match.group(0) if plotly_match else ""
    
    # === STEP 7: Generate final summary section ===
    summary_section_html = _generate_summary_section(final_summary) if final_summary else ""
    
    # === STEP 8: Build complete HTML with ORIGINAL timeline design ===
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
            }}
            .dashboard-container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .dashboard-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .dashboard-header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }}
            .dashboard-content {{
                padding: 40px;
            }}
            .content-section {{
                margin-bottom: 40px;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            .section-title {{
                font-size: 1.8em;
                color: #667eea;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 3px solid #667eea;
            }}
            
            /* Mode Selection Buttons */
            .mode-controls {{
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                justify-content: center;
                flex-wrap: wrap;
            }}
            .mode-btn {{
                padding: 12px 30px;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #667eea;
                background: white;
                color: #667eea;
                border-radius: 25px;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .mode-btn:hover {{
                background: #f0f0ff;
            }}
            .mode-btn.active {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}
            
            /* Entity Selector */
            .entity-selector {{
                text-align: center;
                margin: 20px 0;
                display: none;
            }}
            .entity-selector.active {{
                display: block;
            }}
            .entity-selector select {{
                padding: 10px 20px;
                font-size: 16px;
                border: 2px solid #667eea;
                border-radius: 10px;
                min-width: 300px;
            }}
            
            /* Timeline Navigation */
            .timeline-navigation {{
                text-align: center;
                margin: 30px 0;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 20px;
            }}
            .nav-btn {{
                padding: 10px 20px;
                font-size: 18px;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .nav-btn:hover:not(:disabled) {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }}
            .nav-btn:disabled {{
                opacity: 0.3;
                cursor: not-allowed;
            }}
            .date-range {{
                font-size: 18px;
                font-weight: bold;
                color: #667eea;
                min-width: 200px;
            }}
            
            /* ORIGINAL TIMELINE DESIGN */
            .timeline-wrapper {{
                position: relative;
                padding: 150px 40px;
                min-height: 400px;
            }}
            .timeline-line {{
                position: absolute;
                top: 50%;
                left: 40px;
                right: 40px;
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
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                position: relative;
                z-index: 2;
                cursor: pointer;
                transition: all 0.3s;
            }}
            .point-marker:hover {{
                transform: scale(1.3);
                box-shadow: 0 4px 12px rgba(231, 76, 60, 0.5);
            }}
            .date-label {{
                position: absolute;
                top: -35px;
                left: 50%;
                transform: translateX(-50%);
                font-weight: bold;
                color: #2c3e50;
                font-size: 13px;
                white-space: nowrap;
            }}
            .highlights-box {{
                position: absolute;
                display: none;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                min-width: 280px;
                max-width: 400px;
                border: 2px solid #3498db;
                z-index: 10;
            }}
            .chart-point:hover .highlights-box {{
                display: block;
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
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0 8px 0;
                padding-bottom: 5px;
                border-bottom: 2px solid #3498db;
                font-size: 14px;
            }}
            .region-header:first-child {{
                margin-top: 0;
            }}
            .highlight-item {{
                margin: 8px 0;
                padding: 8px;
                background: #ecf0f1;
                border-radius: 4px;
                font-size: 13px;
                line-height: 1.5;
                color: #34495e;
            }}
            .company-summary {{
                margin: 8px 0;
                padding: 10px;
                background: #ecf0f1;
                border-radius: 4px;
                font-size: 13px;
                line-height: 1.6;
                color: #34495e;
                max-height: 200px;
                overflow-y: auto;
            }}
            
            /* Entity Reports Styles */
            .entity-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }}
            .entity-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.4em;
            }}
            .entity-card p {{
                color: #555;
                line-height: 1.8;
                text-align: justify;
            }}
            .region-section {{
                margin-bottom: 40px;
            }}
            .region-title {{
                font-size: 1.5em;
                color: #764ba2;
                margin-bottom: 20px;
                padding: 15px;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                border-radius: 10px;
            }}
            .summary-box {{
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                padding: 30px;
                border-radius: 10px;
                border-left: 5px solid #764ba2;
                margin-top: 20px;
            }}
            .summary-box p {{
                color: #333;
                line-height: 1.8;
                text-align: justify;
                font-size: 1.05em;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="dashboard-header">
                <h1>📊 Interactive Timeline Dashboard</h1>
                <p>Explore narrative developments across time, entities, and regions</p>
            </div>
            
            <div class="dashboard-content">
                
                {summary_section_html}
                
                <div class="content-section">
                    <div class="section-title">⏱️ Interactive Timeline</div>
                    
                    <!-- Mode Selection -->
                    <div class="mode-controls">
                        <button class="mode-btn active" onclick="switchMode('countries')">🌍 Countries</button>
                        <button class="mode-btn" onclick="switchMode('companies')">🏢 Companies</button>
                        <button class="mode-btn" onclick="switchMode('people')">👥 People</button>
                    </div>
                    
                    <!-- Entity Selectors -->
                    <div id="countrySelector" class="entity-selector active">
                        <select id="countrySelect" onchange="renderTimeline()">
                            {' '.join([f'<option value="{c}">{c}</option>' for c in available_countries])}
                        </select>
                    </div>
                    
                    <div id="companySelector" class="entity-selector">
                        <select id="companySelect" onchange="renderTimeline()">
                            {' '.join([f'<option value="{c}">{c}</option>' for c in available_companies])}
                        </select>
                    </div>
                    
                    <div id="peopleSelector" class="entity-selector">
                        <select id="peopleSelect" onchange="renderTimeline()">
                            {' '.join([f'<option value="{p}">{p}</option>' for p in available_people])}
                        </select>
                    </div>
                    
                    <!-- Timeline Navigation -->
                    <div class="timeline-navigation">
                        <button class="nav-btn" id="prevBtn" onclick="navigatePrev()">← Previous</button>
                        <span class="date-range" id="dateRange"></span>
                        <button class="nav-btn" id="nextBtn" onclick="navigateNext()">Next →</button>
                    </div>
                    
                    <!-- Timeline Display -->
                    <div class="timeline-wrapper" id="timelineContainer">
                        <div class="timeline-line"></div>
                        <div id="timelinePoints"></div>
                    </div>
                </div>
                
                <div class="content-section">
                    <div class="section-title">📈 Interactive Data Analysis</div>
                    {plotly_div}
                </div>
                
            </div>
            
            {entities_html}
            
        </div>
        
        <script>
            // Data
            const countriesData = {json.dumps(countries_timeline_data)};
            const companiesData = {json.dumps(company_summaries_data)};
            const peopleData = {json.dumps(people_summaries_data)};
            
            // State
            let currentMode = 'countries';
            let allDates = [];
            let currentStartIndex = 0;
            const DAYS_TO_SHOW = 7;
            
            // Initialize
            function init() {{
                allDates = Object.keys(countriesData).sort();
                renderTimeline();
            }}
            
            // Mode Switching
            function switchMode(mode) {{
                currentMode = mode;
                currentStartIndex = 0; // Reset to start when switching modes
                
                // Update buttons
                document.querySelectorAll('.mode-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                // Update selectors
                document.querySelectorAll('.entity-selector').forEach(sel => {{
                    sel.classList.remove('active');
                }});
                
                if (mode === 'countries') {{
                    document.getElementById('countrySelector').classList.add('active');
                    // Get all dates from countries data
                    allDates = Object.keys(countriesData).sort();
                }} else if (mode === 'companies') {{
                    document.getElementById('companySelector').classList.add('active');
                    const company = document.getElementById('companySelect').value;
                    if (companiesData[company]) {{
                        allDates = Object.keys(companiesData[company].date_content).sort();
                    }}
                }} else if (mode === 'people') {{
                    document.getElementById('peopleSelector').classList.add('active');
                    const person = document.getElementById('peopleSelect').value;
                    if (peopleData[person]) {{
                        allDates = peopleData[person].dates.sort();
                    }}
                }}
                
                renderTimeline();
            }}
            
            // Navigation
            function navigatePrev() {{
                if (currentStartIndex > 0) {{
                    currentStartIndex = Math.max(0, currentStartIndex - DAYS_TO_SHOW);
                    renderTimeline();
                }}
            }}
            
            function navigateNext() {{
                if (currentStartIndex + DAYS_TO_SHOW < allDates.length) {{
                    currentStartIndex = Math.min(allDates.length - DAYS_TO_SHOW, currentStartIndex + DAYS_TO_SHOW);
                    renderTimeline();
                }}
            }}
            
            // Render Timeline
            function renderTimeline() {{
                const visibleDates = allDates.slice(currentStartIndex, currentStartIndex + DAYS_TO_SHOW);
                const timelinePoints = document.getElementById('timelinePoints');
                timelinePoints.innerHTML = '';
                
                if (visibleDates.length === 0) {{
                    timelinePoints.innerHTML = '<div style="text-align:center; padding:50px;">No data available</div>';
                    updateNavButtons();
                    return;
                }}
                
                // Render based on mode
                if (currentMode === 'countries') {{
                    renderCountriesTimeline(visibleDates);
                }} else if (currentMode === 'companies') {{
                    renderCompaniesTimeline(visibleDates);
                }} else if (currentMode === 'people') {{
                    renderPeopleTimeline(visibleDates);
                }}
                
                updateNavButtons();
                updateDateRange(visibleDates);
            }}
            
            function renderCountriesTimeline(dates) {{
                const country = document.getElementById('countrySelect').value;
                const timelinePoints = document.getElementById('timelinePoints');
                const totalDays = dates.length;
                
                dates.forEach((dateStr, index) => {{
                    const position = (index / Math.max(totalDays - 1, 1)) * 100;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    
                    const highlights = countriesData[dateStr][country] || [];
                    
                    let html = `
                        <div class="chart-point ${{positionClass}}" style="left: ${{position}}%;">
                            <div class="point-marker"></div>
                            <div class="date-label">${{dateDisplay}}</div>
                            <div class="highlights-box">
                                <div class="region-header">${{country}} Companies</div>
                    `;
                    
                    if (highlights.length > 0) {{
                        highlights.forEach(h => {{
                            html += `<div class="highlight-item">${{h}}</div>`;
                        }});
                    }} else {{
                        html += '<div class="highlight-item">No highlights for this date</div>';
                    }}
                    
                    html += '<div class="arrow"></div></div></div>';
                    
                    timelinePoints.innerHTML += html;
                }});
            }}
            
            function renderCompaniesTimeline(dates) {{
                const company = document.getElementById('companySelect').value;
                const companyData = companiesData[company];
                const timelinePoints = document.getElementById('timelinePoints');
                const totalDays = dates.length;
                
                if (!companyData) {{
                    timelinePoints.innerHTML = '<div style="text-align:center; padding:50px;">No data for selected company</div>';
                    return;
                }}
                
                dates.forEach((dateStr, index) => {{
                    const position = (index / Math.max(totalDays - 1, 1)) * 100;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    const content = companyData.date_content[dateStr] || 'No summary available';
                    
                    const html = `
                        <div class="chart-point ${{positionClass}}" style="left: ${{position}}%;">
                            <div class="point-marker"></div>
                            <div class="date-label">${{dateDisplay}}</div>
                            <div class="highlights-box">
                                <div class="region-header">${{company}}</div>
                                <div class="company-summary">${{content}}</div>
                                <div class="arrow"></div>
                            </div>
                        </div>
                    `;
                    
                    timelinePoints.innerHTML += html;
                }});
            }}
            
            function renderPeopleTimeline(dates) {{
                const person = document.getElementById('peopleSelect').value;
                const personData = peopleData[person];
                const timelinePoints = document.getElementById('timelinePoints');
                const totalDays = dates.length;
                
                if (!personData) {{
                    timelinePoints.innerHTML = '<div style="text-align:center; padding:50px;">No data for selected person</div>';
                    return;
                }}
                
                dates.forEach((dateStr, index) => {{
                    const position = (index / Math.max(totalDays - 1, 1)) * 100;
                    const isAbove = index % 2 === 0;
                    const positionClass = isAbove ? 'above' : 'below';
                    const dateDisplay = new Date(dateStr).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                    const dayData = personData.date_content[dateStr];
                    
                    let content = '';
                    if (dayData) {{
                        if (dayData.summary) {{
                            content += `<strong>Summary:</strong><br>${{dayData.summary}}`;
                        }}
                        if (dayData.highlights) {{
                            if (content) content += '<br><br>';
                            content += `<strong>Highlights:</strong><br>${{dayData.highlights}}`;
                        }}
                        if (!content) {{
                            content = 'No data available for this date';
                        }}
                    }} else {{
                        content = 'No data available for this date';
                    }}
                    
                    const html = `
                        <div class="chart-point ${{positionClass}}" style="left: ${{position}}%;">
                            <div class="point-marker"></div>
                            <div class="date-label">${{dateDisplay}}</div>
                            <div class="highlights-box">
                                <div class="region-header">${{person}}</div>
                                <div class="company-summary">${{content}}</div>
                                <div class="arrow"></div>
                            </div>
                        </div>
                    `;
                    
                    timelinePoints.innerHTML += html;
                }});
            }}
            
            function updateNavButtons() {{
                document.getElementById('prevBtn').disabled = (currentStartIndex === 0);
                document.getElementById('nextBtn').disabled = (currentStartIndex + DAYS_TO_SHOW >= allDates.length);
            }}
            
            function updateDateRange(visibleDates) {{
                if (visibleDates.length === 0) {{
                    document.getElementById('dateRange').textContent = 'No dates';
                    return;
                }}
                
                const startDate = new Date(visibleDates[0]);
                const endDate = new Date(visibleDates[visibleDates.length - 1]);
                
                const startStr = startDate.toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}});
                const endStr = endDate.toLocaleDateString('en-US', {{month: 'short', day: 'numeric', year: 'numeric'}});
                
                document.getElementById('dateRange').textContent = `${{startStr}} - ${{endStr}}`;
            }}
            
            // Initialize on load
            init();
        </script>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive timeline dashboard generated: {output_file}")
    return output_file

