# NarrativeMiner

## Automated Narrative Analysis and Mining Tool

**MIGRATION NOTE**: This project has been migrated from `bigdata-client` and `bigdata-research-tools` (including the deprecated `NarrativeMiner` workflow) to REST API + `bigdata-smart-batching`. An LLM mindmap expands a main theme into sub-themes with 2–3 search labels each; search runs on those leaves and all counts roll up at the **sub-theme** level.

This project provides comprehensive narrative analysis and mining capabilities for financial and investment research. It's designed for analysts, portfolio managers, and investment professionals to systematically extract, analyze, and track narrative patterns across various data sources and market segments.

## Features

- **LLM mindmap** — expand a main theme (e.g. "AI Bubble") into sub-themes and searchable claim labels
- **Narrative extraction** from unstructured data using REST search on mindmap leaves
- **Sub-theme-level counts** — charts, summaries, and insights aggregate by sub-theme (not by leaf label)
- **Narrative evolution** and temporal analysis across news, transcripts, and filings

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Narrative_Miners"
   ```

2. **Set up credentials** ([Bigdata.com API key](https://docs.bigdata.com/api-reference/authentication)):
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your credentials:
     ```
     BIGDATA_API_KEY=your_api_key_here
     OPENAI_API_KEY=your_openai_api_key
     ```

3. **Build and run the Docker container**:
   ```bash
   # Build the Docker image
   docker build -t narrative-miners .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app narrative-miners
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `NarrativeMiner.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

#### Setup and Run

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the project**:
   ```bash
   cd "Narrative_Miners"
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   uv pip install jupyterlab
   ```

   **Note**: Dependencies no longer include `bigdata-client` or `bigdata-research-tools`. Instead, use `bigdata-smart-batching` for search and OpenAI for mindmap generation.

4. **Set up credentials**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your credentials:
     ```
     BIGDATA_API_KEY=your_bigdata_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```
   
   **Migration change**: Use `BIGDATA_API_KEY` (not username/password) for REST API authentication.

5. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```

6. **Open the notebook**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `NarrativeMiner.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Workflow

1. Set `main_theme` (default: `"AI Bubble"`)
2. LLM generates a mindmap: theme → sub-themes → 2–3 search labels each → `main_narratives`
3. REST search runs on flattened search labels across news / transcripts / filings
4. Each hit maps to its parent sub-theme (`Label` column)
5. Visualizations, summaries, and exports report counts at **sub-theme** level

## Project Structure

```
Narrative_Miners/
├── README.md                 # Project documentation
├── NarrativeMiner.ipynb      # Main Jupyter notebook for narrative analysis
├── NarrativeMiner.html       # Exported HTML version of the notebook
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── src/
│   ├── mindmap.py           # LLM mindmap generation + plot helpers
│   ├── search_helper.py     # REST search via bigdata-smart-batching
│   ├── llm_helpers.py       # Model defaults / luna-safe sampling
│   ├── bigdata_rest.py      # REST client + universe helpers
│   └── tool.py              # Visualization and analysis helpers
├── scripts/
│   └── smoke_test.py        # Short mindmap + search smoke test
├── output/                  # Generated analysis outputs
└── .venv/                   # Virtual environment (created during setup)
```

## Key Components

- **NarrativeMiner.ipynb**: Main analysis notebook containing the narrative mining workflow
- **src/mindmap.py**: LLM mindmap generation (`main_theme` → sub-themes → search labels)
- **src/tool.py**: Visualization and analysis helpers (counts by `Label` / sub-theme)
- **output/**: Directory containing Excel files with analysis results across different data sources

## Output Files

The analysis generates several Excel files in the `output/` directory:
- **ai_bubble_narrative_analysis.xlsx**: Comprehensive narrative analysis results
- **ai_bubble_filings.xlsx**: Narrative patterns extracted from SEC filings
- **ai_bubble_transcripts.xlsx**: Narrative analysis from earnings call transcripts
- **ai_bubble_news.xlsx**: Narrative patterns from news articles

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Change `main_theme` in the parameters cell to analyze a different narrative
- Large datasets may require significant processing time
- Results are automatically saved to the `output/` directory
