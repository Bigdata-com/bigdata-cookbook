# AI Revenue Generation Market Analysis

## Automated Analysis of AI Revenue Generation Providers and Users

This project provides comprehensive AI revenue generation ecosystem analysis tools that identify technology providers, track customer adoption patterns, and map provider-customer relationships using unstructured data from news sources. It's designed for technology analysts, infrastructure investors, and industry professionals to transform scattered AI signals into quantified market intelligence.

## Features

- **Dual-role classification** distinguishing companies developing AI revenue generation solutions from those implementing them
- **Technology ecosystem mapping** revealing relationships between solution providers and corporate users
- **Adoption timeline tracking** showing how AI revenue generation implementation evolves across different companies
- **Market positioning analysis** quantifying each company's role and exposure in the AI revenue generation ecosystem

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
   cd "AI_Revenue_Generation_Market_Analysis"
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
   - Open `AI_Revenue_Generation_Market_Analysis.py`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
AI_Revenue_Generation_Market_Analysis/
├── README.md                                    # Project documentation
├── AI_Revenue_Generation_Market_Analysis.py     # Main Python script for AI revenue generation analysis
├── requirements.txt                             # Python dependencies
├── .env.example                                # Example environment variables
├── src/
│   ├── tool.py                                 # Core analysis functions and visualization tools
│   ├── search.py                               # Data collection and search functionality
│   ├── network_analysis.py                     # Network analysis and relationship mapping
│   └── labeling.py                             # Classification and labeling functions
└── .venv/                                      # Virtual environment (created during setup)
```

## Key Components

- **AI_Revenue_Generation_Market_Analysis.py**: Main analysis script containing the AI revenue generation ecosystem analysis workflow
- **src/tool.py**: Core Python module with analysis functions and visualization tools
- **src/search.py**: Data collection and search functionality for AI revenue generation related data
- **src/network_analysis.py**: Network analysis and relationship mapping capabilities
- **src/labeling.py**: Classification and labeling functions for provider vs. user identification

## Analysis Features

The AI revenue generation market analysis provides:
- **Provider vs. User Identification**: Automatically categorizes companies based on their role in the AI revenue generation value chain
- **Technology Ecosystem Mapping**: Reveals relationships between solution providers and corporate users
- **Adoption Timeline Tracking**: Shows temporal evolution of AI revenue generation implementation across companies
- **Market Positioning Analysis**: Quantifies company exposure and positioning in the ecosystem

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The script should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook with interactive visualizations
- The tool includes async compatibility setup for Jupyter environments
- Large datasets may require significant processing time 