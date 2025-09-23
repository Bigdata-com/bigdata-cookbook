import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display

html_content = """
<div class="extract-key-insights">
    <h1>Extract Key Insights</h1>

    <div class="intro">
        The visualizations reveal how supply chain reshaping serves as an overarching strategic initiative that
        companies are implementing through various technological and operational sub-themes.
    </div>

    <div class="card-layout">
        <div class="card primary">
            <h3 class="insight-title">AI and Machine Learning Emerges as the Core Enabler</h3>
            <p>With the highest cumulative score across all companies, AI and
            Machine Learning is the most dominant theme, highlighting its
            foundational role in predictive analytics, automation, and
            optimization within modern supply chains.</p>
        </div>

        <div class="card secondary">
            <h3 class="insight-title">Circular Economy and Automation as Structural Shifts</h3>
            <p>The strong presence of Circular Economy Practices and Automation &
            Robotics indicates a structural shift toward sustainable and
            efficient supply chain models—companies are not just digitizing but
            rethinking operational design.</p>
        </div>

        <div class="card tertiary">
            <h3 class="insight-title">Tech-Centric Players Lead the Pack</h3>
            <p>Siemens AG, Infineon Technologies AG, and Qualcomm Inc. are the
            frontrunners in thematic exposure, underscoring that companies at
            the intersection of industrial technology and digital
            infrastructure are best positioned to drive—and benefit from—supply
            chain transformation.</p>
        </div>

        <div class="card quaternary">
            <h3 class="insight-title">IoT Integration as a Bridge Between Physical and Digital</h3>
            <p>IoT's high ranking shows its critical role in connecting assets,
            enabling real-time visibility, and facilitating advanced
            automation, especially for manufacturers and hardware-driven firms.</p>
        </div>
    </div>

    <div class="industry-polarisation-section">
        <div class="polarisation-header">Industry Polarisation</div>

        <div class="sector-engagement">
            <h3 class="section-header">Sector Engagement</h3>
            <ul>
                <li><strong>Semiconductors and Computer Services</strong> industries show the strongest average exposure, reflecting their integral role in enabling supply chain tech (e.g., sensors, connectivity, software).</li>
                <li><strong>Traditional Sectors</strong> like Diversified Industrials show broader but shallower engagement, suggesting they are still in earlier phases of thematic adoption.</li>
            </ul>
        </div>

        <div class="strategic-focus">
            <h3 class="section-header">Strategic Focus</h3>
            <p><strong>Concentration vs. Diversification in Exposure</strong></p>
            <p>Most companies exhibit thematic concentration, focusing efforts on a
            few high-impact areas rather than spreading across all themes—likely
            reflecting strategic prioritization rather than lack of alignment.</p>
        </div>
    </div>
</div>

<style>
/* Main layout and typography */
.extract-key-insights {
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

.extract-key-insights h1 {
    text-align: center;
    margin-bottom: 30px;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
}

.intro {
    margin-bottom: 1.5em;
    font-size: 1.1em;
    line-height: 1.5;
}

/* Card Grid Layout - Force 2x2 grid */
.card-layout {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(2, auto);
    gap: 2rem;
    margin-bottom: 3rem;
}

.card {
    padding: 2.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    font-size: 0.95em;
    line-height: 1.5;
}

/* Card Colors */
.primary {
    background: linear-gradient(135deg, #6fa8dc, #a4c2f4);
    color: white;
}
.secondary {
    background: linear-gradient(135deg, #6aa84f, #b6d7a8);
    color: white;
}
.tertiary {
    background: linear-gradient(135deg, #a64d79, #d5a6bd);
    color: white;
}
.quaternary {
    background: linear-gradient(135deg, #e69138, #f9cb9c);
    color: white;
}

.insight-title {
    font-size: 1.3em;
    margin-top: 0;
    margin-bottom: 1.2em;
    line-height: 1.3;
}

/* Industry Polarisation Section */
.industry-polarisation-section {
    margin-top: 3rem;
    width: 100%;
}

.polarisation-header {
    font-size: 1.4em;
    margin: 1em 0 0.8em 0;
    border-bottom: 2px solid #666;
    padding-bottom: 0.5em;
    font-weight: bold;
}

.section-header {
    font-size: 1.2em;
    margin: 1.5em 0 1em 0;
    font-weight: bold;
}

.sector-engagement {
    padding: 0 15px;
    margin-bottom: 2em;
}

.sector-engagement ul {
    margin-left: 1em;
    padding-left: 1em;
}

.sector-engagement li {
    margin-bottom: 1.2em;
    list-style-type: disc;
    padding-left: 0.5em;
}

.strategic-focus {
    padding: 0 15px;
}

/* Responsive adjustments */
@media (max-width: 600px) {
    .card-layout {
        grid-template-columns: 1fr;
    }
}
</style>
"""

def create_thematic_exposure_dashboard(df_company, n_companies=10, interactive=True):
    """
    Creates five separate figures for analyzing thematic exposure of companies.
    
    Args:
        df_company: DataFrame with company thematic exposure data
        n_companies: Number of top companies to analyze
        interactive: If True, creates interactive Plotly plots. If False, creates static matplotlib plots.
    """
    # Select top n companies and reset index
    df = df_company[:n_companies].reset_index(drop=True).copy()

    # Extract theme column names (all columns between 'Industry' and 'Composite Score')
    theme_columns = list(df.iloc[:, 3:-1].columns)

    # Create all figures
    fig_heatmap = create_company_theme_heatmap(df, theme_columns, interactive)
    fig_total_scores = create_company_scores_bar_chart(df, interactive)
    fig_scatter_themes = create_top_themes_scatter_plot(df, theme_columns, interactive)
    fig_bar_themes = create_themes_summary_bar_chart(df, theme_columns, interactive)
    fig_industry_heatmap = create_industry_analysis_heatmap(df, theme_columns, interactive)

    return fig_heatmap, fig_total_scores, fig_scatter_themes, fig_bar_themes, fig_industry_heatmap


def create_company_theme_heatmap(df, theme_columns, interactive=True):
    """
    Creates a heatmap showing thematic exposure scores for each company.
    """
    if interactive:
        # Original Plotly version
        fig = go.Figure()
        heatmap_z = df[theme_columns].values
        heatmap_x = theme_columns
        heatmap_y = df['Company'].tolist()

        fig.add_trace(
            go.Heatmap(
                z=heatmap_z,
                x=heatmap_x,
                y=heatmap_y,
                colorscale='YlGnBu',
                text=heatmap_z.astype(int),
                texttemplate="%{text}",
                showscale=True,
            )
        )

        fig.update_layout(
            title='Company Thematic Exposure Heatmap (Raw Scores)',
            height=600,
            width=1200,
            margin=dict(l=60, r=50, t=80, b=50),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Company")
        )

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        # Create heatmap data
        heatmap_data = df[theme_columns].values
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   xticklabels=theme_columns,
                   yticklabels=df['Company'].tolist(),
                   annot=True,
                   fmt='d',
                   cmap='YlGnBu',
                   cbar=True)
        
        plt.title('Company Thematic Exposure Heatmap (Raw Scores)', fontsize=16, pad=20)
        plt.xlabel('Themes', fontsize=12)
        plt.ylabel('Company', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_company_scores_bar_chart(df, interactive=True):
    """
    Creates a horizontal bar chart showing total composite scores for companies.
    """
    companies = df['Company'].tolist()
    total_scores = df['Composite Score'].tolist()

    # Sort by score for better visualization (highest first)
    sorted_indices = np.argsort(total_scores)[::-1]
    sorted_companies = [companies[i] for i in sorted_indices]
    sorted_scores = [total_scores[i] for i in sorted_indices]
    
    if interactive:
        # Original Plotly version
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=sorted_companies,
                x=sorted_scores,
                orientation='h',
                marker=dict(
                    color=sorted_scores,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=sorted_scores,
                textposition='outside',
                textfont=dict(size=10),
            )
        )

        fig.update_layout(
            title='Company Total Composite Scores',
            height=600,
            width=1200,
            margin=dict(l=60, r=50, t=80, b=50),
            xaxis=dict(title="Total Composite Score"),
            yaxis=dict(title="Company")
        )

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_scores)))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(sorted_companies)), sorted_scores, color=colors)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            plt.text(bar.get_width() + max(sorted_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.0f}', va='center', fontsize=10)
        
        plt.yticks(range(len(sorted_companies)), sorted_companies)
        plt.xlabel('Total Composite Score', fontsize=12)
        plt.ylabel('Company', fontsize=12)
        plt.title('Company Total Composite Scores', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()


def create_top_themes_scatter_plot(df, theme_columns, interactive=True):
    """
    Creates a scatter plot showing the top 3 thematic exposures for each company.
    """
    if interactive:
        # Original Plotly version
        fig = go.Figure()

        max_score = df[theme_columns].values.max()
        companies_unique = df['Company'].unique()

        for i, company in enumerate(companies_unique):
            company_data = df[df['Company'] == company]
            if len(company_data) == 0:
                continue

            company_row = company_data.iloc[0]
            company_scores = company_row[theme_columns].values
            top_indices = np.argsort(company_scores)[-3:]

            x_values = []
            y_values = []
            sizes = []
            hover_texts = []

            for idx in top_indices:
                if company_scores[idx] > 0:
                    theme = theme_columns[idx]
                    score = company_scores[idx]
                    size = (score / max_score) * 80
                    x_values.append(company)
                    y_values.append(theme)
                    sizes.append(size)
                    hover_texts.append(f"{company}<br>{theme}: {int(score)}")

            if len(x_values) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            sizemode='area',
                            sizeref=0.15,
                            color=i,
                            colorscale='Turbo',
                            showscale=False,
                            opacity=0.7,
                            line=dict(width=1, color='DarkSlateGrey'),
                        ),
                        text=hover_texts,
                        hoverinfo='text',
                        name=company,
                    )
                )

        fig.update_layout(
            height=600,
            width=1200,
            title_text="Top 3 Thematic Exposures by Company",
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=50),
        )

        fig.update_xaxes(title_text="Company")
        fig.update_yaxes(title_text="Theme")

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        max_score = df[theme_columns].values.max()
        companies_unique = df['Company'].unique()
        
        # Create color map
        colors = plt.cm.turbo(np.linspace(0, 1, len(companies_unique)))
        
        # Collect all data points
        all_x = []
        all_y = []
        all_sizes = []
        all_colors = []
        
        for i, company in enumerate(companies_unique):
            company_data = df[df['Company'] == company]
            if len(company_data) == 0:
                continue

            company_row = company_data.iloc[0]
            company_scores = company_row[theme_columns].values
            top_indices = np.argsort(company_scores)[-3:]

            for idx in top_indices:
                if company_scores[idx] > 0:
                    theme = theme_columns[idx]
                    score = company_scores[idx]
                    size = (score / max_score) * 500  # Scale for matplotlib
                    
                    all_x.append(company)
                    all_y.append(theme)
                    all_sizes.append(size)
                    all_colors.append(colors[i])
        
        # Create scatter plot
        plt.scatter(all_x, all_y, s=all_sizes, c=all_colors, alpha=0.7, edgecolors='darkslategray', linewidth=1)
        
        plt.xlabel('Company', fontsize=12)
        plt.ylabel('Theme', fontsize=12)
        plt.title('Top 3 Thematic Exposures by Company', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_themes_summary_bar_chart(df, theme_columns, interactive=True):
    """
    Creates a horizontal bar chart showing total scores across all themes.
    """
    # Calculate theme totals across all companies
    theme_totals = df[theme_columns].sum()
    theme_names = theme_totals.index.tolist()
    theme_values = theme_totals.values.tolist()

    # Sort by values (descending)
    sorted_indices = np.argsort(theme_values)[::-1]
    sorted_themes = [theme_names[i] for i in sorted_indices]
    sorted_values = [theme_values[i] for i in sorted_indices]
    
    if interactive:
        # Original Plotly version
        fig = go.Figure()

        # Create custom color scale that never goes to white
        normalized_values = np.array(sorted_values)
        normalized_values = (normalized_values - normalized_values.min()) / (normalized_values.max() - normalized_values.min())
        # Scale to range 0.3 to 1.0 to avoid white colors
        color_values = 0.3 + (normalized_values * 0.7)

        fig.add_trace(
            go.Bar(
                y=sorted_themes,
                x=sorted_values,
                orientation='h',
                marker=dict(
                    color=color_values,
                    colorscale=[[0, '#8B0000'], [1, '#FF4500']],  # Dark red to orange-red, no white
                    showscale=False
                ),
                text=sorted_values,
                textposition='outside',
                textfont=dict(size=10),
            )
        )

        fig.update_layout(
            height=600,
            width=1200,
            title_text="Total Thematic Scores Summary",
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=50),
        )

        fig.update_xaxes(title_text="Total Score Across All Companies")
        fig.update_yaxes(title_text="Theme")

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        # Create custom color scale (dark red to orange-red)
        normalized_values = np.array(sorted_values)
        normalized_values = (normalized_values - normalized_values.min()) / (normalized_values.max() - normalized_values.min())
        # Scale to range 0.3 to 1.0 to avoid white colors
        color_values = 0.3 + (normalized_values * 0.7)
        
        # Create custom colormap
        colors = plt.cm.Reds(color_values)
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(sorted_themes)), sorted_values, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_values)):
            plt.text(bar.get_width() + max(sorted_values) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0f}', va='center', fontsize=10)
        
        plt.yticks(range(len(sorted_themes)), sorted_themes)
        plt.xlabel('Total Score Across All Companies', fontsize=12)
        plt.ylabel('Theme', fontsize=12)
        plt.title('Total Thematic Scores Summary', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()


def create_industry_analysis_heatmap(df, theme_columns, interactive=True):
    """
    Creates a heatmap showing average thematic scores grouped by industry.
    """
    # Group by industry and calculate mean scores
    industry_data = []

    for industry, group in df.groupby('Industry'):
        for theme in theme_columns:
            industry_data.append({
                'Industry': industry,
                'Theme': theme,
                'Average_Score': group[theme].mean()
            })

    industry_df = pd.DataFrame(industry_data)

    # Create a pivot table for the heatmap
    industry_pivot = industry_df.pivot(index='Industry', columns='Theme', values='Average_Score')

    if interactive:
        # Original Plotly version
        fig = go.Figure(data=go.Heatmap(
            z=industry_pivot.values,
            x=industry_pivot.columns,
            y=industry_pivot.index,
            colorscale='YlOrRd',
            text=np.round(industry_pivot.values, 1),
            texttemplate="%{text}",
            showscale=True,
        ))

        # Format the industry analysis figure
        fig.update_layout(
            title='Industry-Level Thematic Exposure Analysis (Average Scores)',
            height=500,
            width=1200,
            margin=dict(l=60, r=50, t=80, b=50),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Industry")
        )

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 6))
        
        # Create heatmap
        sns.heatmap(industry_pivot, 
                   annot=True,
                   fmt='.1f',
                   cmap='YlOrRd',
                   cbar=True)
        
        plt.title('Industry-Level Thematic Exposure Analysis (Average Scores)', fontsize=16, pad=20)
        plt.xlabel('Themes', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()
    
def display_figures(df_company, interactive=True, n_companies=10):
    """Helper function to create and display figures based on type"""
    
    # Create figures with the specified interactive parameter
    figures = create_thematic_exposure_dashboard(df_company, n_companies=n_companies, interactive=interactive)
    
    # Display figures based on type
    for fig in figures:
        if interactive and hasattr(fig, 'show'):
            fig.show()
        else:
            plt.show()


def display_figures_cookbooks(df_company, interactive=True, n_companies=10):
    figures = create_thematic_exposure_dashboard(df_company, n_companies=n_companies, interactive=interactive)
    # Check if running on Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    if interactive and is_colab:
        for i, fig in enumerate(figures):
            html_str = fig.to_html(
                include_plotlyjs='cdn', 
                div_id=f"plot_{i}",
                config={'displayModeBar': True}
            )
            
            styled_html = f"""
            <div style="margin: 10px 0; padding: 5px;">
                {html_str}
            </div>
            """
            display(HTML(styled_html))
    elif interactive and not is_colab:
        display_figures(df_company, interactive=interactive)
    else:
        for i, fig in enumerate(figures):
            display(fig)  # Use display() instead of plt.show()
            plt.close()
            
            # Add space between plots
            if i < len(figures) - 1:
                print("\n")