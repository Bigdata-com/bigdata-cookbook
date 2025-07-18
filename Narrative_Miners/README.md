# NarrativeMiner

## Automated Narrative Analysis and Mining Tool

This project provides comprehensive narrative analysis and mining capabilities for financial and investment research. It's designed for analysts, portfolio managers, and investment professionals to systematically extract, analyze, and track narrative patterns across various data sources and market segments.

## Features

- **Narrative extraction** and pattern recognition from unstructured data
- **Sentiment analysis** and narrative sentiment tracking
- **Narrative evolution** and temporal analysis
- **Automated narrative scoring** and ranking systems

## Local Installation and Usage

### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup and Run

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

4. **Set up credentials**:
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

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

1. **Clone and navigate to the project**:
   ```bash
   cd "Narrative_Miners"
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

3. **Build the Docker image**:
   ```bash
   docker build -t narrative-miner .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 narrative-miner
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `NarrativeMiner.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t narrative-miner .

# Run the container with port mapping
docker run -p 8888:8888 narrative-miner

# Run in background
docker run -d -p 8888:8888 --name narrative-miner-container narrative-miner

# Stop the container
docker stop narrative-miner-container

# Remove the container
docker rm narrative-miner-container

# View logs
docker logs narrative-miner-container

# Access container shell
docker exec -it narrative-miner-container bash

# Remove the image
docker rmi narrative-miner
```

## Project Structure

```
Narrative_Miners/
├── README.md                 # Project documentation
├── NarrativeMiner.ipynb      # Main Jupyter notebook for narrative analysis
├── NarrativeMiner.html       # Exported HTML version of the notebook
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Docker ignore file
├── .env.example             # Example environment variables
├── src/
│   └── tool.py              # Core narrative mining functionality
├── output/                  # Generated analysis outputs
│   ├── ai_bubble_narrative_analysis.xlsx
│   ├── ai_bubble_filings.xlsx
│   ├── ai_bubble_transcripts.xlsx
│   └── ai_bubble_news.xlsx
└── .venv/                   # Virtual environment (created during setup)
```

## Key Components

- **NarrativeMiner.ipynb**: Main analysis notebook containing the narrative mining workflow
- **src/tool.py**: Core Python module with narrative extraction and analysis functions
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
- Large datasets may require significant processing time
- Results are automatically saved to the `output/` directory