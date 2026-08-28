# Rising Bond Spread Risks

> **SDK Migration**: This project no longer uses `bigdata-client` or `bigdata-research-tools`. Use `src/bigdata_rest.py` and `src/search_helper.py` with your `BIGDATA_API_KEY`. Company universes are CSV files (RP_ENTITY_ID, COMPANY_NAME). Advanced features like entity sentiment and risk tree generation require custom implementation.

## Analyzing Spillover Risks from Rising Bond Spreads in Western Europe

This workflow identifies and quantifies sovereign exposure to bond market vulnerabilities and financial contagion across Western European nations. It uses Bigdata Research Tools to create a risk taxonomy, retrieve and label news content, calculate country-level risk exposure, and generate interactive dashboards with AI-powered narrative summaries.

## Features

- Risk taxonomy generation with LLM-powered mind mapping
- Country-level risk scoring across sub-scenarios
- Rolling sentiment indicators (30/90 days) and volume tracking
- Interactive dashboards with gauge indicators and time series
- AI-powered narrative summaries at peak coverage moments
- CSV exports for further analysis

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
```bash
cd "Rising_Bond_Spread_Risks"
```

2. **Set up credentials** ([Bigdata.com API key](https://docs.bigdata.com/api-reference/authentication)):
- Copy the example environment file:
```bash
cp .env.example .env
```
- Edit the `.env` file and add your credentials:
```
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_KEY=your_openai_api_key
```

3. **Build and run the Docker container**:
```bash
# Build the Docker image
docker build -t bond-spread-risks .

# Run the container
docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app bond-spread-risks
```

4. **Access JupyterLab**:
- Open your browser and navigate to `http://localhost:8888`
- Open `Rising_Bond_Spread_Risks.ipynb`
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
cd "Rising_Bond_Spread_Risks"
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
BIGDATA_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_KEY=your_openai_api_key
```

6. **Start JupyterLab**:
```bash
jupyter lab
```

7. **Open the notebook**:
- When the server starts, it will display one or more access URLs in the terminal
- Click a provided URL (or copy/paste into your browser) to open JupyterLab
- Open `Rising_Bond_Spread_Risks.ipynb`
- Follow the setup instructions in the notebook
- Run cells sequentially to perform the analysis

## Project Structure

```
Rising_Bond_Spread_Risks/
├── README.md                           # Project documentation
├── Rising_Bond_Spread_Risks.ipynb      # Main Jupyter notebook
├── requirements.txt                    # Python dependencies
├── .env.example                        # Example environment variables
└── src/                                # Risk analysis modules
    ├── entity_risk_prompt_labeler.py   # Risk tree generation and labeling
    ├── narrative_dashboard.py          # Interactive dashboard generation
    ├── report_generator.py             # AI narrative summarization
    ├── search_entities.py              # Bigdata search orchestration
    ├── sentiment_analysis.py           # Rolling sentiment indicators
    └── visualization_tool.py           # Cross-country risk comparison charts
```

## Key Components

- **Rising_Bond_Spread_Risks.ipynb**: Main notebook containing the complete risk analysis workflow
- **src/entity_risk_prompt_labeler.py**: Generates risk taxonomy and labels content relevance
- **src/sentiment_analysis.py**: Calculates rolling sentiment and volume indicators
- **src/narrative_dashboard.py**: Produces interactive dashboards with time series and narratives
- **src/visualization_tool.py**: Creates comparative risk visualizations across countries

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Results (CSV and PNG files) are written into the `output/` folder
- Graphviz installation is required for taxonomy visualization


