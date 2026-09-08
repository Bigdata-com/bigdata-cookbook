# Daily Digest Crude Oil

## Automated Crude Oil Market Monitoring and Analysis Tool

This project provides comprehensive monitoring and analysis of crude oil market trends and news. It's designed for commodity traders, energy analysts, and investment professionals to systematically track, analyze, and understand market-moving narratives in the crude oil sector and their potential price impacts.

## Features

- **Lexicon generation** of crude oil industry-specific terminology and jargon
- **Real-time content retrieval** via Bigdata API with parallelized keyword searches
- **Topic clustering and selection** with AI-powered verification and ranking
- **Custom report generation** with configurable ranking systems for trending topics
- **Market impact assessment** scoring topics for trendiness, novelty, and magnitude

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Daily_Digest_Crude_Oil"
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
   docker build -t daily-digest-crude-oil .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app daily-digest-crude-oil
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Daily_Digest_Crude_Oil.ipynb`
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
   cd "Daily_Digest_Crude_Oil"
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   uv pip install jupyterlab
   ```

4. **Set up credentials**:
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

5. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```

6. **Open the notebook**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `Daily_Digest_Crude_Oil.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Daily_Digest_Crude_Oil/
├── README.md                      # Project documentation
├── Daily_Digest_Crude_Oil.ipynb  # Main Jupyter notebook for crude oil analysis
├── Daily_Digest_Crude_Oil.html   # Exported HTML version of the notebook
├── requirements.txt              # Python dependencies
├── .env.example                 # Example environment variables
├── Dockerfile                   # Docker configuration
├── src/
│   ├── lexicon_generator.py     # Industry-specific terminology generation
│   ├── search_topics.py         # Content retrieval and search functionality
│   ├── topics_extractor.py      # Topic clustering and analysis
│   └── report_generator.py      # Custom report generation
├── assets/                      # Report template assets
│   ├── bigdata-logo-white.svg
│   ├── report_template.html
│   ├── arrow_up.png
│   ├── arrow_down.png
│   ├── arrow_mid.png
│   └── flame-icon.png
├── report/                      # Generated daily reports
│   ├── 2025-06-25_Crude_Oil.html
│   └── daily_digest_crude_oil.csv
└── .venv/                      # Virtual environment (created during setup)
```

## Key Components

- **Daily_Digest_Crude_Oil.ipynb**: Main analysis notebook implementing the four-step agentic workflow
- **src/lexicon_generator.py**: Generates crude oil industry-specific terminology for comprehensive news retrieval
- **src/search_topics.py**: Handles content retrieval via Bigdata API with parallelized searches
- **src/topics_extractor.py**: AI-powered topic clustering, verification, and ranking
- **src/report_generator.py**: Custom HTML report generation with market impact scoring

## Analysis Workflow

The notebook implements a comprehensive four-step workflow:

1. **Lexicon Generation**: Creates industry-specific jargon and terminology to maximize recall in news retrieval
2. **Content Retrieval**: Uses Bigdata API to fetch relevant content with daily windows and parallelized keyword searches
3. **Topic Clustering & Selection**: Verifies, groups, and summarizes news into ranked trending topics with AI scoring
4. **Custom Report Generation**: Produces daily digest reports with configurable ranking and granular source verification

## Use Cases

- **Commodity Trading**: Monitor supply disruptions, geopolitical tensions, and production changes
- **Energy Market Analysis**: Track OPEC decisions, inventory reports, and refinery operations
- **Price Forecasting**: Identify emerging narratives that could influence crude oil prices
- **Risk Management**: Assess geopolitical and operational risks affecting oil markets
- **Investment Strategy**: Support energy sector investment decisions with timely market intelligence

## Sample Analysis

The workflow includes a practical example tracking crude oil market developments during the Israel-Iran tensions of June 2025, demonstrating how to transform unstructured news into structured, ranked insights on market-moving narratives.

## Market Areas Covered

- **Supply & Demand**: Production changes, inventory levels, consumption patterns
- **Geopolitical Events**: Regional conflicts, sanctions, diplomatic developments
- **OPEC & Policy**: Production decisions, quota changes, policy announcements
- **Infrastructure**: Pipeline disruptions, refinery operations, transportation issues
- **Economic Indicators**: Global economic conditions affecting oil demand

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results and daily reports are automatically saved to respective directories
- Custom scoring parameters can be modified in the notebook configuration for different market focuses
- Reports include both summary insights and granular news sources for verification
- Consider running analysis during market hours for most relevant real-time insights
