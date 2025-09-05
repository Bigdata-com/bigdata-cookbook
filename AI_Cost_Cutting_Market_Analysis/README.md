# AI Cost Cutting Market Analysis

## Automated Analysis of AI Cost Cutting Providers and Users

This project provides comprehensive AI cost cutting ecosystem analysis tools that identify technology providers, track customer adoption patterns, and map provider-customer relationships using unstructured data from news sources. It's designed for technology analysts, investors, and industry professionals to transform scattered AI signals into quantified market intelligence and identify investment opportunities in the AI cost cutting value chain.

## Features

- **Dual-role classification** distinguishing companies developing AI cost cutting solutions from those implementing them
- **Technology ecosystem mapping** revealing relationships between solution providers and corporate users
- **Adoption timeline tracking** showing how AI cost cutting implementation evolves across different sectors
- **Market positioning analysis** quantifying each company's role and exposure in the AI cost cutting ecosystem

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "AI_Cost_Cutting_Market_Analysis"
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
   docker build -t ai_cost_cutting_market_analysis .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app ai_cost_cutting_market_analysis

   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `AI_Cost_Cutting_Market_Analysis.ipynb`
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
   cd "AI_Cost_Cutting_Market_Analysis"
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
   - Open `AI_Cost_Cutting_Market_Analysis.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
AI_Cost_Cutting_Market_Analysis/
├── README.md                                    # Project documentation
├── AI_Cost_Cutting_Market_Analysis.ipynb       # Main Jupyter notebook for AI cost cutting analysis
├── AI_Cost_Cutting_Market_Analysis.html        # Exported HTML version of the notebook
├── requirements.txt                            # Python dependencies
├── .env.example                               # Example environment variables
├── src/
│   ├── tool.py                               # Core analysis functions and visualization tools
│   ├── search.py                             # Data collection and search functionality
│   ├── network_analysis.py                   # Network analysis and relationship mapping
│   └── labeling.py                           # Classification and labeling functions
└── .venv/                                    # Virtual environment (created during setup)
```

## Key Components

- **AI_Cost_Cutting_Market_Analysis.ipynb**: Main analysis notebook containing the AI cost cutting ecosystem analysis workflow
- **src/tool.py**: Core Python module with analysis functions and visualization tools
- **src/search.py**: Data collection and search functionality for AI cost cutting related data
- **src/network_analysis.py**: Network analysis and relationship mapping capabilities
- **src/labeling.py**: Classification and labeling functions for provider vs. user identification

## Analysis Features

The AI cost cutting market analysis provides:
- **Provider vs. User Identification**: Automatically categorizes companies based on their role in the AI cost cutting value chain
- **Technology Ecosystem Mapping**: Reveals relationships between AI solution providers and enterprise users
- **Adoption Timeline Tracking**: Shows temporal evolution of AI cost cutting implementation across sectors
- **Market Positioning Analysis**: Quantifies company exposure and positioning in the AI cost cutting ecosystem
- **Sector-Specific Insights**: Identifies key players across Financial Services, Technology, Consumer, and Industrial sectors

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook with interactive visualizations
- The tool includes async compatibility setup for Jupyter environments
- Large datasets may require significant processing time
- Weekly aggregation provides cleaner trend analysis compared to daily granularity 