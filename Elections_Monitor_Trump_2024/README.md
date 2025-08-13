# Trump Reelection Impact Analysis

## Automated Analysis of Corporate Perspectives on Trump's Presidential Re-election

This project provides comprehensive tools for analyzing corporate positioning toward Trump reelection impact using unstructured data from executive transcripts. It's designed for analysts, portfolio managers, and investment professionals to transform scattered executive commentary into quantified political exposure metrics and identify investment opportunities based on corporate positioning analysis.

## Features

- **Positive vs. negative impact assessment** distinguishing companies that expect benefits from those anticipating challenges under Trump policies
- **Sector-wide political exposure mapping** revealing industry patterns in Trump administration positioning
- **Temporal exposure tracking** showing how political expectations evolve over time
- **Corporate-political topic networks** identifying key policy themes and company concerns through relationship analysis
- **Confidence scoring system** quantifying the balance between positive and negative Trump impact mentions


## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system


#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Elections_Monitor_Trump_2024"
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
   docker build -t trump-reelections-impact-analysis .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app trump-reelections-impact-analysis
   ```


4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Trump_Reelection_Impact_Analisys.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis


### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup and Run

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the project**:
   ```bash
   cd "Elections_Monitor_Trump_2024"
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
   - Open `Trump_Reelection_Impact_Analysis.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis



## Project Structure

```
Elections_Monitor_Trump_2024/
├── README.md                      # Project documentation
├── Trump_Reelection_Impact_Analysis.ipynb  # Main Jupyter notebook for Trump impact analysis
├── Trump_Reelection_Impact_Analysis.html   # Exported HTML version of the notebook
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment variables
├── src/
│   ├── tool.py                    # Core analysis and visualization functions
│   ├── search.py                  # Search functionality for retrieving relevant content
│   ├── labeling.py                # LLM-based classification functions
│   ├── visualization_tools.py     # Network analysis and visualization tools
│   └── query_builder.py           # Query construction utilities
└── .venv/                         # Virtual environment (created during setup)
```

## Key Components

- **Trump_Reelection_Impact_Analysis.ipynb**: Main analysis notebook containing the complete Trump reelection impact assessment workflow
- **src/tool.py**: Core Python module with analysis functions for plotting, confidence scoring, and basket identification
- **src/search.py**: Search utilities for retrieving relevant executive transcript content from Bigdata API
- **src/labeling.py**: LLM-powered classification for Trump reelection impact assessment
- **src/visualization_tools.py**: Network analysis tools for corporate-political topic relationship mapping

## Analysis Features

The Trump reelection impact analysis provides:
- **Political Positioning Classification**: Uses LLM to categorize executive commentary as positive, negative, or unclear regarding Trump impact
- **Sector-Based Impact Mapping**: Identifies industry patterns in Trump policy expectations
- **Temporal Evolution Tracking**: Monitors how the positioning about Trump reelection develops over time
- **Network Analysis**: Reveals relationships between companies and topics through co-mention analysis


## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook with interactive visualizations
- Custom entity filters and analysis parameters can be modified in the notebook configuration
