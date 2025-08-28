# Tracking Inflation Drivers

## Automated Macroeconomic Inflation Analysis Tool

This project provides comprehensive inflation analysis capabilities for macroeconomic research. It's designed for analysts, portfolio managers, and economic researchers to systematically track and analyze inflation dynamics across different economic drivers and components.

## Features

- **Automated theme breakdown** into specific inflation components and drivers
- **Systematic document analysis** using embeddings-based search and classification
- **Economic categorization** that turns narrative signals into structured insights
- **Comprehensive reporting** with analytical summaries for each inflation driver

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Tracking_Inflation_Drivers"
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
   docker build -t tracking-inflation-drivers .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app tracking-inflation-drivers
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Tracking_Inflation_Drivers.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Graphviz](https://pypi.org/project/graphviz/) - Required for graph visualization features

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
   cd "Tracking_Inflation_Drivers"
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
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `Tracking_Inflation_Drivers.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Tracking_Inflation_Drivers/
├── README.md                        # Project documentation
├── Tracking_Inflation_Drivers.py    # Main Python script for inflation analysis
├── requirements.txt                 # Python dependencies
├── .env.example                    # Example environment variables
├── src/
│   ├── search.py                   # Search functionality
│   ├── token_manager.py            # Token management utilities
│   ├── summarize.py                # Summarization and reporting tools
│   ├── labels.py                   # Labeling and categorization functions
│   └── mind_map_tools.py           # Mind mapping utilities
├── output/                         # Generated reports and analysis results
└── .venv/                          # Virtual environment (created during setup)
```

## Key Components

- **Tracking_Inflation_Drivers.py**: Main analysis script containing the complete inflation tracking workflow
- **src/search.py**: Search functionality for content retrieval across news and documents
- **src/summarize.py**: AI-powered summarization and HTML report generation
- **src/labels.py**: Economic categorization and labeling functions
- **src/mind_map_tools.py**: Theme taxonomy generation and visualization

## Analysis Features

The inflation tracking system provides:
- **Theme Taxonomy Generation**: LLM-powered breakdown of inflation into specific components
- **Semantic Content Retrieval**: Embeddings-based search and document classification
- **Economic Driver Categorization**: Classification of inflation drivers by economic nature
- **Comprehensive Reporting**: Professional analytical reports with insights and visualizations

## Inflation Driver Categories Covered

- **Demand-pull**: When consumer demand exceeds available supply of goods and services
- **Cost-push**: When production costs like labor or raw materials increase
- **Wage increases**: Rising wages contributing to inflation through higher consumer prices
- **Global factors**: International commodity prices and exchange rate impacts
- **Monetary policy**: Central bank actions and interest rate influences

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The script should be run sequentially from top to bottom
- Analysis results are displayed inline and saved to the `output/` directory
- Custom inflation parameters can be modified in the script configuration
- Graphviz installation is required for theme visualization features 