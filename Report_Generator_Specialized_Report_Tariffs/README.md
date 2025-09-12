# Report Generator - Specialized Report Tariffs

## Automated Analysis of Trade Tariff Risks and Corporate Mitigation Strategies

This workflow generates sector-wide and company-specific reports on tariff-related risks. It uses Bigdata Research Tools to create a risk taxonomy, retrieve and label content, extract corporate mitigation plans from filings/transcripts (with optional News fallback), and produce executive and detailed HTML reports with CSV exports.

## Features

- Sector-level thematic summaries and company-level risk summaries
- Extraction of mitigation plans from Filings/Transcripts (optional News fallback)
- Executive and Detailed HTML reports
- CSV exports for further analysis

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
```bash
cd "Report_Generator_Specialized_Report_Tariffs"
```

2. **Set up credentials**:
- Copy the example environment file:
```bash
cp .env.example .env
```
- Edit the `.env` file and add your credentials:
```
BIGDATA_USERNAME=your_username
BIGDATA_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

3. **Build and run the Docker container**:
```bash
# Build the Docker image
docker build -t report-generator-tariffs .

# Run the container
docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app report-generator-tariffs
```

4. **Access JupyterLab**:
- Open your browser and navigate to `http://localhost:8888`
- Open `Report_Generator_Specialized_Report_Tariffs.ipynb`
- Follow the setup instructions in the notebook
- Run cells sequentially to perform the analysis

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Graphviz](https://pypi.org/project/graphviz/) - Required for taxonomy visualization

#### Setup and Run

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Install Graphviz** (required for graph visualization):
```bash
# On macOS
brew install graphviz

# On Ubuntu/Debian
sudo apt-get install graphviz

# On Windows
# Download from https://graphviz.org/download/
```

3. **Clone and navigate to the project**:
```bash
cd "Report_Generator_Specialized_Report_Tariffs"
```

4. **Create a virtual environment and install dependencies**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install jupyterlab
```

5. **Set up credentials**:
- Copy the example environment file:
```bash
cp .env.example .env
```
- Edit the `.env` file and add your credentials:
```
BIGDATA_USERNAME=your_username
BIGDATA_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

6. **Start JupyterLab**:
```bash
jupyter lab
```

7. **Open the notebook**:
- When the server starts, it will display one or more access URLs in the terminal
- Click a provided URL (or copy/paste into your browser) to open JupyterLab
- Open `Report_Generator_Specialized_Report_Tariffs.ipynb`
- Follow the setup instructions in the notebook
- Run cells sequentially to perform the analysis

## Project Structure

```
Report_Generator_Specialized_Report_Tariffs/
├── README.md                                 # Project documentation
├── Report_Generator_Specialized_Report_Tariffs.html   # Exported HTML version of the notebook
├── Report_Generator_Specialized_Report_Tariffs.ipynb  # Main Jupyter notebook
├── requirements.txt                          # Python dependencies
├── .env.example                              # Example environment variables
└── src/                                       # Report generation modules
    ├── html_report.py                         # HTML report builders (executive & detailed)
    ├── report_generator.py                    # Orchestrates summarization, scoring, responses
    ├── label/                                 # Labeling pipeline for documents
    │   ├── label_process.py                   # Labeling process runner
    │   ├── label_prompts.py                   # Prompts used for labeling
    │   └── labels.py                          # Label definitions
    ├── mindmap/                               # Taxonomy/mindmap helpers
    │   ├── generate_trees.py                  # Build themes tree
    │   ├── theme_prompts.py                   # Prompts for theme generation
    │   └── themes.py                          # Theme models and types
    ├── response/                              # Company response extraction
    │   ├── company_response.py                # Extract mitigation plans
    │   └── response_prompts.py                # Prompts for responses
    ├── search/                                # Retrieval helpers
    │   ├── content_retrieval.py               # DataRetriever for Bigdata API
    │   ├── query_tools.py                     # Query utilities
    │   ├── search.py                          # Search orchestration
    │   └── sentences.py                       # Sentence utilities
    └── summary/                               # Summarization and scoring
        ├── summary_prompts.py                 # Prompts for summaries
        ├── summary.py                         # Topic summarizers
        └── token_manager.py                   # Token management utils
```

## Key Components

- **Report_Generator_Specialized_Report_Tariffs.ipynb**: Main notebook containing the report generation workflow
- **src/report_generator.py**: Orchestrates summarization, scoring, and mitigation plan extraction
- **src/html_report.py**: Produces executive and detailed HTML reports

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Results (HTML and CSV) are written into the `output/` folder
- Graphviz installation is required for taxonomy visualization


