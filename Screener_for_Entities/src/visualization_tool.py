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
import warnings
warnings.filterwarnings('ignore')

# Optional imports for unified dashboard approach
try:
    from bigdata_research_tools.visuals.risk_visuals import create_risk_exposure_dashboard
    from bigdata_research_tools.visuals.thematic_visuals import create_thematic_exposure_dashboard
    UNIFIED_VISUALS_AVAILABLE = True
except ImportError:
    UNIFIED_VISUALS_AVAILABLE = False


def create_entity_theme_heatmap(df, theme_columns, interactive=True):
    """
    Creates a heatmap showing thematic exposure scores for each entity.
    """
    if interactive:
        # Original Plotly version
        fig = go.Figure()
        heatmap_z = df[theme_columns].values
        heatmap_x = theme_columns
        heatmap_y = df['Entity'].tolist()

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
            title='Entity Risk Exposure Heatmap (Raw Scores)',
            height=600,
            width=1200,
            margin=dict(l=60, r=50, t=80, b=50),
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(title="Entity")
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
                   yticklabels=df['Entity'].tolist(),
                   annot=True,
                   fmt='d',
                   cmap='YlGnBu',
                   cbar=True)
        
        plt.title('Entity Risk Exposure Heatmap (Raw Scores)', fontsize=16, pad=20)
        plt.xlabel('Risk Sub-Scenarios', fontsize=12)
        plt.ylabel('Entity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_entity_scores_bar_chart(df, interactive=True):
    """
    Creates a horizontal bar chart showing total composite scores for entities.
    """
    entities = df['Entity'].tolist()
    total_scores = df['Composite Score'].tolist()

    # Sort by score for better visualization (highest first)
    sorted_indices = np.argsort(total_scores)[::-1]
    sorted_entities = [entities[i] for i in sorted_indices]
    sorted_scores = [total_scores[i] for i in sorted_indices]
    
    if interactive:
        # Original Plotly version
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=sorted_entities,
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
            title='Total Entity Risk Exposure Score',
            height=600,
            width=1200,
            margin=dict(l=60, r=50, t=80, b=50),
            xaxis=dict(title="Total Risk Exposure Score"),
            yaxis=dict(title="Entity")
        )

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_scores)))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(sorted_entities)), sorted_scores, color=colors)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            plt.text(bar.get_width() + max(sorted_scores) * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.0f}', va='center', fontsize=10)
        
        plt.yticks(range(len(sorted_entities)), sorted_entities)
        plt.xlabel('Total Composite Score', fontsize=12)
        plt.ylabel('Entity', fontsize=12)
        plt.title('Total Entity Composite Score', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()


def create_top_themes_scatter_plot(df, theme_columns, interactive=True):
    """
    Creates a scatter plot showing the top 3 thematic exposures for each entity.
    """
    if interactive:
        # Original Plotly version
        fig = go.Figure()

        max_score = df[theme_columns].values.max()
        entities_unique = df['Entity'].unique()

        for i, entity in enumerate(entities_unique):
            entity_data = df[df['Entity'] == entity]
            if len(entity_data) == 0:
                continue

            entity_row = entity_data.iloc[0]
            entity_scores = entity_row[theme_columns].values
            top_indices = np.argsort(entity_scores)[-3:]

            x_values = []
            y_values = []
            sizes = []
            hover_texts = []

            for idx in top_indices:
                if entity_scores[idx] > 0:
                    theme = theme_columns[idx]
                    score = entity_scores[idx]
                    size = (score / max_score) * 80
                    x_values.append(entity)
                    y_values.append(theme)
                    sizes.append(size)
                    hover_texts.append(f"{entity}<br>{theme}: {int(score)}")

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
                        name=entity,
                    )
                )

        fig.update_layout(
            height=600,
            width=1200,
            title_text="Top Risk Exposures by Entity",
            showlegend=False,
            margin=dict(l=60, r=50, t=80, b=50),
        )

        fig.update_xaxes(title_text="Entity")
        fig.update_yaxes(title_text="Risk Sub-Scenario")

        return fig
    else:
        # Static matplotlib version
        plt.figure(figsize=(15, 8))
        
        max_score = df[theme_columns].values.max()
        entities_unique = df['Entity'].unique()
        
        # Create color map
        colors = plt.cm.turbo(np.linspace(0, 1, len(entities_unique)))
        
        # Collect all data points
        all_x = []
        all_y = []
        all_sizes = []
        all_colors = []
        for i, entity in enumerate(entities_unique):
            entity_data = df[df['Entity'] == entity]
            if len(entity_data) == 0:
                continue

            entity_row = entity_data.iloc[0]
            entity_scores = entity_row[theme_columns].values
            top_indices = np.argsort(entity_scores)[-3:]

            for idx in top_indices:
                if entity_scores[idx] > 0:
                    theme = theme_columns[idx]
                    score = entity_scores[idx]
                    size = (score / max_score) * 500  # Scale for matplotlib

                    all_x.append(entity)
                    all_y.append(theme)
                    all_sizes.append(size)
                    all_colors.append(colors[i])
        
        # Create scatter plot
        plt.scatter(all_x, all_y, s=all_sizes, c=all_colors, alpha=0.7, edgecolors='darkslategray', linewidth=1)
        
        plt.xlabel('Entity', fontsize=12)
        plt.ylabel('Risk Sub-Scenario', fontsize=12)
        plt.title('Top Risk Exposures by Entity', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt.gcf()


def create_themes_summary_bar_chart(df, theme_columns, interactive=True):
    """
    Creates a horizontal bar chart showing total scores across all themes.
    """
    # Calculate theme totals across all entities
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

        fig.update_xaxes(title_text="Total Score Across All Entities")
        fig.update_yaxes(title_text="Risk Sub-Scenario")

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
        plt.xlabel('Total Risk Score Across All Entities', fontsize=12)
        plt.ylabel('Risk Sub-Scenario', fontsize=12)
        plt.title('Risk Scores by Sub-Scenario', fontsize=16, pad=20)
        plt.tight_layout()
        
        return plt.gcf()

def create_thematic_exposure_dashboard(df_entity, n_entities=10, interactive=True):
    """
    Creates five separate figures for analyzing thematic exposure of entities.

    Args:
        df_entity: DataFrame with entity thematic exposure data
        n_entities: Number of top entities to analyze
        interactive: If True, creates interactive Plotly plots. If False, creates static matplotlib plots.
    """
    # Select top n entities and reset index
    df = df_entity[:n_entities].reset_index(drop=True).copy()

    # Extract theme column names (all columns between 'Industry' and 'Composite Score')
    theme_columns = list(df.iloc[:, 3:-1].columns)

    # Create all figures
    fig_heatmap = create_entity_theme_heatmap(df, theme_columns, interactive)
    fig_total_scores = create_entity_scores_bar_chart(df, interactive)
    fig_scatter_themes = create_top_themes_scatter_plot(df, theme_columns, interactive)
    fig_bar_themes = create_themes_summary_bar_chart(df, theme_columns, interactive)

    return fig_heatmap, fig_total_scores, fig_scatter_themes, fig_bar_themes

def display_figures(df_entity, interactive=True, n_entities=10):
    """Helper function to create and display figures based on type"""
    
    # Create figures with the specified interactive parameter
    figures = create_thematic_exposure_dashboard(df_entity, n_entities=n_entities, interactive=interactive)

    # Display figures based on type
    for fig in figures:
        if interactive and hasattr(fig, 'show'):
            fig.show()
        else:
            plt.show()


def display_risk_figures(df_entity, interactive=True, n_entities=10):
    """Helper function to create and display risk figures based on type"""
    
    # Create figures with the specified interactive parameter
    figures = create_risk_exposure_dashboard(df_entity, n_entities=n_entities, interactive=interactive)

    # Display figures based on type
    for fig in figures:
        if interactive:
            fig.show()
        else:
            plt.show()


def display_figures_cookbooks(df_entity, interactive=True, n_entities=10):
    figures = create_thematic_exposure_dashboard(df_entity, n_entities=n_entities, interactive=interactive)
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
        display_figures(df_entity, interactive=interactive)
    else:
        for i, fig in enumerate(figures):
            display(fig)  # Use display() instead of plt.show()
            plt.close()
            
            # Add space between plots
            if i < len(figures) - 1:
                print("\n")

def display_entity_dashboard(df_entity, dashboard_type="thematic", interactive=True, n_entities=10, config=None, for_cookbooks=False):
    """
    Unified function to display entity dashboards (thematic or risk).
    
    Parameters:
    -----------
    df_entity : pandas.DataFrame
        DataFrame containing entity exposure data
    dashboard_type : str, default="thematic"
        Type of dashboard: "thematic" or "risk"
    interactive : bool, default=True
        If True, creates interactive Plotly plots
    n_entities : int, default=10
        Number of top entities to analyze
    config : dict, optional
        Configuration dictionary to customize the dashboard
    for_cookbooks : bool, default=False
        If True, uses cookbook-compatible display (handles Colab, etc.)
    """
    
    if not interactive:
        print("Static matplotlib version not implemented for unified dashboards.")
        print("Falling back to interactive Plotly version.")
        interactive = True
    
    # Check if running on Google Colab
    is_colab = False
    if for_cookbooks:
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
    
    try:
        # Create the dashboard figures
        main_dashboard, country_dashboard = create_unified_entity_dashboard(
            df_entity, n_entities=n_entities, dashboard_type=dashboard_type, config=config
        )
        
        # Set titles based on dashboard type
        if dashboard_type.lower() == "risk":
            figure_titles = [
                "Entity Risk Exposure Dashboard",
                "Country-Level Risk Analysis"
            ]
        else:
            figure_titles = [
                "Entity Thematic Exposure Dashboard", 
                "Country-Level Thematic Analysis"
            ]
        
        figures = [main_dashboard, country_dashboard]
        
        if interactive and for_cookbooks and is_colab:
            # Special handling for Google Colab
            for i, (fig, title) in enumerate(zip(figures, figure_titles)):
                print(f"\n{title}:")
                html_str = fig.to_html(
                    include_plotlyjs='cdn',
                    div_id=f"{dashboard_type}_plot_{i}",
                    config={'displayModeBar': True}
                )
                
                styled_html = f"""
                <div style="margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h3 style="margin-top: 0;">{title}</h3>
                    {html_str}
                </div>
                """
                display(HTML(styled_html))
                
        elif interactive:
            # Standard display
            for title, fig in zip(figure_titles, figures):
                print(f"\n{title}:")
                fig.show()
        else:
            # Non-interactive fallback
            for i, (fig, title) in enumerate(zip(figures, figure_titles)):
                print(f"\n{title}:")
                display(fig)
                
                if i < len(figures) - 1:
                    print("\n")
                    
    except Exception as e:
        print(f"Error creating {dashboard_type} dashboard: {e}")
        print("Make sure your DataFrame has the required columns:")
        print("- Entity")
        print("- Country")
        print("- Composite Score") 
        print(f"- {dashboard_type.title()} exposure columns (numeric)")


# ===== SIMPLIFIED UNIFIED APPROACH =====

def display_risk_figures(df_entity, interactive=True, n_entities=10, config=None):
    """
    Display risk dashboard for entities using the simple, working approach.
    Uses existing thematic functions with risk-appropriate titles.
    
    Parameters:
    -----------
    df_entity : pandas.DataFrame
        DataFrame with Entity, Country, Composite Score, and risk exposure columns
    interactive : bool, default=True
        If True, creates interactive Plotly plots
    n_entities : int, default=10
        Number of top entities to analyze
    config : dict, optional
        Configuration dictionary (currently unused but kept for API consistency)
    """
    
    # Create dashboard using existing working functions
    figures = create_thematic_exposure_dashboard(df_entity, n_entities=n_entities, interactive=interactive)
    
    # Display with risk-focused titles
    risk_titles = [
        'Entity Risk Exposure Heatmap (Raw Scores)',
        'Total Entity Risk Exposure Score', 
        'Top Risk Exposures by Entity',
        'Risk Exposure Scores across Sub-Scenarios'
    ]
    
    for i, fig in enumerate(figures):
        if hasattr(fig, 'update_layout') and i < len(risk_titles):
            fig.update_layout(title=risk_titles[i])
            fig.show()
        else:
            print(f"Risk Analysis Chart {i+1}:")
            if hasattr(fig, 'show'):
                fig.show()
            else:
                display(fig)


def create_risk_exposure_dashboard(df_entity, n_entities=10, interactive=True):
    """
    Creates four separate figures for analyzing risk exposure of entities.
    Now that the underlying chart functions have risk-appropriate titles, 
    this simply calls the same functions as create_thematic_exposure_dashboard.
    """
    # Select top n entities and reset index
    df = df_entity[:n_entities].reset_index(drop=True).copy()

    # Extract risk column names (all columns between 'Industry' and 'Composite Score')
    risk_columns = list(df.iloc[:, 3:-1].columns)

    # Create all figures - titles are now risk-focused in the underlying functions
    fig_heatmap = create_entity_theme_heatmap(df, risk_columns, interactive)
    fig_total_scores = create_entity_scores_bar_chart(df, interactive)
    fig_scatter_risks = create_top_themes_scatter_plot(df, risk_columns, interactive)
    fig_bar_risks = create_themes_summary_bar_chart(df, risk_columns, interactive)

    return fig_heatmap, fig_total_scores, fig_scatter_risks, fig_bar_risks


def display_risk_figures_cookbooks(df_entity, interactive=True, n_entities=10, config=None):
    """
    Display risk dashboard for entities (cookbook compatible).
    Uses risk-focused dashboard with proper titles.
    """
    # Check if running on Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    # Create dashboard using risk-focused functions  
    figures = create_risk_exposure_dashboard(df_entity, n_entities=n_entities, interactive=interactive)
    
    # Risk-focused titles (for fallback)
    risk_titles = [
        'Entity Risk Exposure Heatmap (Raw Scores)',
        'Total Entity Risk Exposure Score',
        'Top Risk Exposures by Entity', 
        'Risk Exposure Scores across Sub-Scenarios'
    ]
    
    if interactive and is_colab:
        # Special handling for Google Colab
        for i, (fig, title) in enumerate(zip(figures, risk_titles)):
            if hasattr(fig, 'update_layout'):
                fig.update_layout(title=title)
                
            html_str = fig.to_html(
                include_plotlyjs='cdn',
                div_id=f"risk_plot_{i}",
                config={'displayModeBar': True}
            )
            
            styled_html = f"""
            <div style="margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                <h3 style="margin-top: 0;">{title}</h3>
                {html_str}
            </div>
            """
            display(HTML(styled_html))
    else:
        # Standard display
        for i, (fig, title) in enumerate(zip(figures, risk_titles)):
            if hasattr(fig, 'update_layout'):
                fig.update_layout(title=title)
            
            if hasattr(fig, 'show'):
                fig.show()
            else:
                display(fig)


def display_thematic_figures(df_entity, interactive=True, n_entities=10, config=None):
    """Display thematic dashboard for entities"""
    display_figures(df_entity, interactive, n_entities)

def display_thematic_figures_cookbooks(df_entity, interactive=True, n_entities=10, config=None):
    """Display thematic dashboard for entities (cookbook compatible)"""
    display_figures_cookbooks(df_entity, interactive, n_entities)

