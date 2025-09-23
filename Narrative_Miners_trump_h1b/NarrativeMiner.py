# %% [markdown]
# # Narrative Miners: Uncover the Stories That Drive Markets
# 
# ## Automated Analysis of Market Narratives Across Multiple Document Sources

# %% [markdown]
# ## Why It Matters
# 
# Understanding how market narratives emerge and evolve across different information sources is crucial for investment decision-making, but manually tracking narrative development across scattered news coverage, earnings calls, and regulatory filings is time consuming. Investment decisions need systematic analysis of narrative progression to identify emerging trends and timing patterns.
# 
# ## What It Does
# 
# The `NarrativeMiner` class in the bigdata-research-tools package systematically tracks narrative evolution across multiple document types using unstructured data from news, transcripts, and filings. Built for analysts, portfolio managers, and investment professionals, it transforms scattered narrative signals into quantified trend intelligence and identifies timing patterns across different information sources.
# 
# ## How It Works
# 
# The `NarrativeMiner` combines **multi-source content retrieval**, **temporal narrative tracking**, and **cross-source comparative analysis** to deliver:
# 
# - **Cross-document narrative mapping** across news media, earnings calls, and SEC filings
# - **Temporal evolution tracking** showing how narratives develop and change over time across sources
# - **Intensity measurement** quantifying narrative prevalence and significance across document types
# 
# ## A Real-World Use Case
# 
# This cookbook demonstrates the complete workflow through analyzing "AI Bubble Concerns" narrative as it emerges and evolves across news, earnings calls, and regulatory filings, highlighting the difference between public discourse and corporate communications.

# %% [markdown]
# ## Setup and Imports

# %% [markdown]
# ## Async Compatibility Setup
# 
# **Run this cell first** - Required for Google Colab, Jupyter Notebooks, and VS Code with Jupyter extension:
# 
# ### Why is this needed?
# 
# Interactive environments (Colab, Jupyter) already have an asyncio event loop running. When bigdata-research-tools makes async API calls (like to OpenAI), you'll get this error without nest_asyncio:
# 
# ```
# RuntimeError: asyncio.run() cannot be called from a running event loop
# ```
# 
# The `nest_asyncio.apply()` command patches this to allow nested event loops.
# 
# 💡 **Tip**: If you're unsure which environment you're in, just run the cell below - it won't hurt in any environment!

# %%
import datetime
start = datetime.datetime.now()

try:
    import asyncio
    asyncio.get_running_loop()
    import nest_asyncio; nest_asyncio.apply()
    print("✅ nest_asyncio applied")
except (RuntimeError, ImportError):
    print("✅ nest_asyncio not needed or not available")

# %% [markdown]
# ## Environment Setup
# 
# The following cell configures the necessary path for the analysis

# %%
import os
import sys

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
print(f"✅ Local environment setup complete")

# %% [markdown]
# ## Import Required Libraries
# 
# Import the core libraries needed for narrative mining analysis, including the custom visualization and analysis tools.

# %%
from IPython.display import display, HTML, IFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

from src.tool import (
    load_results, 
    extract_narrative_insights,
    create_source_summary,
    display_sample_data,
    visualize_cross_source_narratives, 
    visualize_news_narrative_breakdown
)

from bigdata_research_tools.workflows.narrative_miner import NarrativeMiner
from bigdata_research_tools.excel import ExcelManager
from bigdata_client import Bigdata
from bigdata_client.daterange import RollingDateRange
from bigdata_client.models.sources import Source
from bigdata_client.models.search import DocumentType

# %% [markdown]
# ## Optional: Plotly Display Configuration
# 
# For better visualization rendering, you can also set the Plotly renderer:

# %%
import plotly
import plotly.graph_objects as go
import plotly.io as pio

# Try to detect the environment and set appropriate renderer
try:
    # Check if we're in JupyterLab
    import os
    if 'JUPYTERHUB_SERVICE_PREFIX' in os.environ or 'JPY_SESSION_NAME' in os.environ:
        pio.renderers.default = 'jupyterlab'
        print("✅ Plotly configured for JupyterLab")
    else:
        # Default for VS Code, Jupyter Notebook, etc.
        pio.renderers.default = 'plotly_mimetype+notebook'
        print("✅ Plotly configured for Jupyter/VS Code")
except:
    # Fallback to a more universal renderer
    pio.renderers.default = 'notebook'
    print("✅ Plotly configured with fallback renderer")



# %% [markdown]
# ## Define Output Paths
# 
# We define the output paths for our narrative mining results.

# %%
# Define output file paths for our results
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

news_results_path = f"{output_dir}/ai_bubble_news.xlsx"
transcripts_results_path = f"{output_dir}/ai_bubble_transcripts.xlsx"
filings_results_path = f"{output_dir}/ai_bubble_filings.xlsx"
visualization_path = f"{output_dir}/ai_bubble_narratives.html"

# %% [markdown]
# ## Load Credentials

# %%
from dotenv import load_dotenv
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
load_dotenv(script_dir / '.env')

BIGDATA_USERNAME = os.getenv('BIGDATA_USERNAME')
BIGDATA_PASSWORD = os.getenv('BIGDATA_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not all([BIGDATA_USERNAME, BIGDATA_PASSWORD, OPENAI_API_KEY]):
    print("❌ Missing required environment variables")
    raise ValueError("Missing required environment variables. Check your .env file.")
else:
    print("✅ Credentials loaded from .env file")

# %% [markdown]
# ## Connecting to Bigdata
# 
# Create a Bigdata object with your credentials.

# %%
bigdata = Bigdata(BIGDATA_USERNAME, BIGDATA_PASSWORD)

# %% [markdown]
# ## Defining the Narrative Analysis Parameters
# 
# ### Fixed Parameters
# - **AI Bubble Narratives** (`main_narratives`): Specific narrative sentences related to AI bubble concerns
# - **Common Parameters** (`common_params`): Shared configuration across all narrative miners
# - **Model Selection** (`llm_model`): The LLM model used for narrative labeling and analysis
# - **Time Period** (`start_date` and `end_date`): The date range over which to run the analysis
# - **Rerank Threshold** (`rerank_threshold`): Cross-encoder threshold for result relevance filtering
# - **Document Limits** (`document_limit`): Maximum number of documents to retrieve per query
# - **Search Frequency** (`freq`): Frequency of date ranges for search operations

# %%
# AI Bubble Narratives
main_narratives = [
    "Tech valuations have detached from fundamental earnings potential",
    "AI investments show classic signs of irrational exuberance",
    "Market is positioning AI as revolutionary without proven ROI",
    "Current AI investments may not generate predicted financial returns",
    "Tech CEOs acknowledge AI implementation challenges amid high expectations",
    "Analysts are questioning the timeline for AI-driven profits",
    "Companies are spending billions on unproven AI technology",
    "AI infrastructure costs are rising but revenue gains remain uncertain",
    "Venture capital is flooding AI startups at unsustainable valuations",
    "Regulatory concerns could derail AI market growth projections",
    "Public discourse about AI capabilities exceeds technical realities",
    "AI talent acquisition costs have created an unsustainable bubble",
    "Corporate executives privately express concerns about AI ROI timelines",
    "AI market projections rely on aggressive and unproven assumptions",
    "Industry veterans drawing parallels to previous tech bubbles"
]

# LLM Specification
llm_model = "openai::gpt-4o-mini"


# Specify Time Range
start_date = "2024-03-01"
end_date = "2025-03-28"

# Rerank Threshold
rerank_threshold = 0.7

# Search Frequency
freq = '6M'

# Fiscal Year
fiscal_year = 2024

# Document Limits
document_limit = 10

# Commen Params
common_params = {
    "narrative_sentences": main_narratives,
    "llm_model": llm_model,
    "start_date": start_date,
    "end_date": end_date,
    "rerank_threshold": rerank_threshold}

# %% [markdown]
# ## Configure the Narrative Miners
# 
# Create narrative miners for each document type. In this example, we select MT Newswires as the news source.

# %%
# Common Params
common_params = {
    "narrative_sentences": main_narratives,
    "llm_model": llm_model,
    "start_date": start_date,
    "end_date": end_date,
    "rerank_threshold": rerank_threshold}
    
# Choose MT Newswires as a news source
tech_news_sources = bigdata.knowledge_graph.find_sources("MT Newswires")
tech_news_ids = [source.id for source in tech_news_sources if "MT Newswires" == source.name]

# Create the specialized miners for each document type
news_miner = NarrativeMiner(
    sources=tech_news_ids,
    document_type=DocumentType.NEWS,
    fiscal_year=None,
    **common_params
)

transcripts_miner = NarrativeMiner(
    sources=None,
    document_type=DocumentType.TRANSCRIPTS,
    fiscal_year=fiscal_year,
    **common_params
)

filings_miner = NarrativeMiner(
    sources=None,
    fiscal_year=fiscal_year,
    document_type=DocumentType.FILINGS,
    **common_params
)

# %% [markdown]
# ## Run Narrative Mining Across Sources
# 
# Execute the narrative mining processes for news, earnings call transcripts, and SEC filings:

# %%
# Mine news narratives
print("Mining news narratives...")
try:
    news_results = news_miner.mine_narratives(
        document_limit=document_limit,
        freq=freq,
        export_path=news_results_path
    )
    print("✅ News mining completed successfully!")
except Exception as e:
    print(f"Warning during news mining: {e}")

# %%
# Mine transcripts narratives
print("Mining earnings call transcripts...")
try:
    transcripts_results = transcripts_miner.mine_narratives(
        document_limit=document_limit,
        freq=freq,
        export_path=transcripts_results_path
    )
    print("✅ Transcripts mining completed successfully!")
except Exception as e:
    print(f"Warning during transcripts mining: {e}")

# %%
# Mine filings narratives
print("Mining SEC filings...")
try:
    filings_results = filings_miner.mine_narratives(
        document_limit=document_limit,
        freq=freq,
        export_path=filings_results_path
    )
    print("✅ Filings mining completed successfully!")
except Exception as e:
    print(f"Warning during filings mining: {e}")

# %% [markdown]
# ## Load and Process Results
# 
# Load the exported Excel files, clean the data, and display a summary.

# %%
# Load results from all three document types with labeling
news_df = load_results(news_results_path, "News Media")
transcripts_df = load_results(transcripts_results_path, "Earnings Calls")
filings_df = load_results(filings_results_path, "SEC Filings")

# Create and display summary
source_summary = create_source_summary(news_df, transcripts_df, filings_df)
display(source_summary)

# Display sample data from each source
display_sample_data(news_df, transcripts_df, filings_df)

# %% [markdown]
# ## Create Narrative Visualizations
# 
# Generate comparative visualizations showing narrative evolution across sources and detailed breakdown of news narratives.

# %%
news_df.head(2)

# %%
import warnings
warnings.filterwarnings("ignore", message=".*'method'.*", category=FutureWarning)

fig1 = visualize_cross_source_narratives(news_df, transcripts_df, filings_df, interactive=False) #set interactive=True (or False) to enable (or disable) the interactive plot
fig1.show()  

# %%
# Create the narrative breakdown visualization for news 
fig2 = visualize_news_narrative_breakdown(news_df, interactive=False) #set interactive=True (or False) to enable (or disable) the interactive plot
fig2.show()

# %% [markdown]
# ## Extract and Display Key Insights
# 
# Extract key insights from the narrative mining data and display them.

# %%
# Extract insights from our narrative mining data
insights = extract_narrative_insights(news_df, transcripts_df, filings_df)

print("## AI Bubble Narrative Key Insights\n")
print(f"Peak month for news coverage: {insights['peak_news_month']}")
print(f"Peak month for earnings call mentions: {insights['peak_transcript_month']}")
print(f"Peak month for regulatory filing mentions: {insights['peak_filing_month']}")
print(f"\nDominant narrative in news: \"{insights['top_news_narrative']}\"")
print(f"Dominant narrative in earnings calls: \"{insights['top_transcript_narrative']}\"")
print(f"Dominant narrative in regulatory filings: \"{insights['top_filing_narrative']}\"")
print(f"\nTotal narrative mentions in news: {insights['total_news_mentions']}")
print(f"Total mentions in earnings calls: {insights['total_transcript_mentions']}")
print(f"Total mentions in regulatory filings: {insights['total_filing_mentions']}")
print(f"\nAverage lag between news coverage peaks and SEC filings: {insights['avg_lag_days']} days")

# %% [markdown]
# ## Export the Results
# 
# Export the data as Excel files for further analysis or to share with the team.

# %%
try:
    # Create the Excel manager
    excel_manager = ExcelManager()

    # Define the dataframes and their sheet configurations
    df_args = [
        (news_df, "News Narratives", (0, 0)),
        (transcripts_df, "Earnings Call Narratives", (0, 0)),
        (filings_df, "SEC Filing Narratives", (0, 0)),
        (source_summary, "Summary", (1, 1))
    ]

    # Save the workbook
    combined_results_path = f"{output_dir}/ai_bubble_narrative_analysis.xlsx"
    excel_manager.save_workbook(df_args, combined_results_path)
    
    print(f"✅ Results exported to {combined_results_path}")

except Exception as e:
    print(f"Warning while exporting to excel: {e}")

# %% [markdown]
# ## Conclusion
# 
# The Narrative Miners reveal important patterns in how the AI bubble narrative evolved across information sources:
# 
# **Timing and Intensity Variations:**
# 
# - News media shows major spikes in AI bubble concerns, often leading the narrative cycle with the highest peaks
# - Earnings calls demonstrate cyclical attention to bubble concerns, with executives addressing topics most prominently during specific quarters
# - SEC filings show the most volatile pattern with multiple significant spikes, suggesting ongoing regulatory concerns
# 
# **Narrative Progression:**
# 
# - Media coverage often leads the initial bubble narrative, potentially triggering corporate responses visible in earnings calls
# - Corporate executives' discussions peak during specific periods but tend to diminish over time
# - SEC filing mentions frequently show increased intensity throughout the analysis period, indicating persistent regulatory attention
# 
# **Cross-Source Intelligence:**
# 
# - Different sources provide complementary perspectives on the same underlying narrative
# - Timing lags between sources reveal information flow patterns and decision-making hierarchies
# - The intensity patterns help identify when narratives are gaining or losing momentum across different stakeholder groups
# 
# This analysis demonstrates how systematic narrative mining across multiple document types provides richer insights than analyzing any single source in isolation.


