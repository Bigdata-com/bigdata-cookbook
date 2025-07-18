import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def display_figures(df_company, interactive=True, n_companies=10):
    """Helper function to create and display figures based on type"""
    
    # Create figures with the specified interactive parameter
    fig_heatmap, fig_total_scores, fig_scatter_themes, fig_bar_themes, fig_industry_heatmap = create_thematic_exposure_dashboard(
        df_company, n_companies=n_companies, interactive=interactive
    )
    
    figures = [fig_heatmap, fig_total_scores, fig_scatter_themes, fig_bar_themes, fig_industry_heatmap]
    figure_names = ["Risk Heatmap", "Total Scores", "Scatter Themes", "Bar Themes", "Industry Heatmap"]
    
    # Display figures based on type
    for fig, name in zip(figures, figure_names):
        if interactive and hasattr(fig, 'show'):
            fig.show()
        else:
            plt.show()


def create_thematic_exposure_dashboard(df_company, n_companies=10, interactive=True):
    """
    Creates five separate figures for analyzing thematic exposure of companies.

    Parameters:
    -----------
    df_company : pandas.DataFrame
        DataFrame containing company data with columns for 'Company', 'Industry',
        'Composite Score', and multiple thematic exposure columns.
    n_companies : int, default=10
        Number of companies to include in the analysis.
    interactive : bool, default=True
        If True, creates interactive Plotly plots. If False, creates static matplotlib plots.

    Returns:
    --------
    tuple
        A tuple containing five figures:
        - fig_heatmap: Company-theme exposure heatmap
        - fig_total_scores: Total composite scores bar chart
        - fig_scatter_themes: Top themes scatter plot by company
        - fig_bar_themes: Thematic scores horizontal bar chart
        - fig_industry_heatmap: Industry-level analysis heatmap
    """
    # Select top n companies and reset index
    df = df_company[:n_companies].reset_index(drop=True).copy()

    # Extract theme column names (all columns between 'Industry' and 'Composite Score')
    theme_columns = list(df.iloc[:, 4:-1].columns)

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

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame containing company and theme data
    theme_columns : list
        List of column names representing thematic categories
    interactive : bool, default=True
        If True, creates interactive Plotly plot. If False, creates static matplotlib plot.

    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Heatmap figure with color scale
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
            title='Risk Exposure Heatmap (Raw Scores)',
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
        
        plt.title('Risk Exposure Heatmap (Raw Scores)', fontsize=16, pad=20)
        plt.xlabel('Sub-Scenarios', fontsize=12)
        plt.ylabel('Company', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_company_scores_bar_chart(df, interactive=True):
    """
    Creates a horizontal bar chart showing total composite scores for companies.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame containing company scores
    interactive : bool, default=True
        If True, creates interactive Plotly plot. If False, creates static matplotlib plot.

    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Bar chart figure showing total composite scores
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
            title='Total Risk Exposure Score',
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
        plt.title('Total Risk Exposure Score', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()


def create_top_themes_scatter_plot(df, theme_columns, interactive=True):
    """
    Creates a scatter plot showing the top 3 thematic exposures for each company.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame containing company and theme data
    theme_columns : list
        List of column names representing thematic categories
    interactive : bool, default=True
        If True, creates interactive Plotly plot. If False, creates static matplotlib plot.

    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Scatter plot showing top themes by company
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
            title_text="Top Risk Exposures by Company",
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=50),
        )

        fig.update_xaxes(title_text="Company")
        fig.update_yaxes(title_text="Sub-Scenario")

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
        plt.ylabel('Sub-Scenario', fontsize=12)
        plt.title('Top Risk Exposures by Company', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_themes_summary_bar_chart(df, theme_columns, interactive=True):
    """
    Creates a horizontal bar chart showing total scores across all themes.
    Uses a red color palette that never goes to white.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame containing company and theme data
    theme_columns : list
        List of column names representing thematic categories
    interactive : bool, default=True
        If True, creates interactive Plotly plot. If False, creates static matplotlib plot.

    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Horizontal bar chart showing thematic scores summary
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
            title_text="Risk Exposure Scores across Sub-Scenarios",
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=50),
        )

        fig.update_xaxes(title_text="Total Score Across Companies")
        fig.update_yaxes(title_text="Sub-Scenario")

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
        plt.xlabel('Total Score Across Companies', fontsize=12)
        plt.ylabel('Sub-Scenario', fontsize=12)
        plt.title('Risk Exposure Scores across Sub-Scenarios', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()


def create_industry_analysis_heatmap(df, theme_columns, interactive=True):
    """
    Creates a heatmap showing average thematic scores grouped by industry.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame containing company and industry data
    theme_columns : list
        List of column names representing thematic categories
    interactive : bool, default=True
        If True, creates interactive Plotly plot. If False, creates static matplotlib plot.

    Returns:
    --------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        Heatmap figure showing industry-level thematic analysis
    """
    # Group by industry and calculate mean scores
    industry_data = []

    for industry, group in df.groupby('Industry'):
        for theme in theme_columns:
            industry_data.append({
                'Industry': industry,
                'Sub-Scenario': theme,
                'Average_Score': group[theme].mean()
            })

    industry_df = pd.DataFrame(industry_data)

    # Create a pivot table for the heatmap
    industry_pivot = industry_df.pivot(index='Industry', columns='Sub-Scenario', values='Average_Score')

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
            title='Industry-Level Risk Exposure (Average Scores)',
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
        
        plt.title('Industry-Level Risk Exposure (Average Scores)', fontsize=16, pad=20)
        plt.xlabel('Sub-Scenarios', fontsize=12)
        plt.ylabel('Industry', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()