# Liquid Cooling Market Watch

## Automated Analysis of Liquid Cooling Technology Providers and Adopters

This project provides comprehensive liquid cooling ecosystem analysis tools that identify technology providers, track customer adoption patterns, and map provider-customer relationships using unstructured data from news sources. It's designed for technology analysts, infrastructure investors, and industry professionals to transform scattered technology signals into quantified market intelligence.

## Features

- **Dual-role classification** distinguishing companies developing liquid cooling solutions from those implementing them
- **Technology ecosystem mapping** revealing relationships between solution providers and data center operators
- **Adoption timeline tracking** showing how liquid cooling implementation evolves across different sectors
- **Market positioning analysis** quantifying each company's role and exposure in the liquid cooling ecosystem

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
   cd "Liquid_Cooling_Market_Watch"
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
   - Open `Liquid_Cooling_Market_Watch.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Liquid_Cooling_Market_Watch/
├── README.md                           # Project documentation
├── Liquid_Cooling_Market_Watch.ipynb   # Main Jupyter notebook for liquid cooling analysis
├── Liquid_Cooling_Market_Watch.html    # Exported HTML version of the notebook
├── requirements.txt                    # Python dependencies
├── .env.example                       # Example environment variables
├── src/
│   ├── tool.py                        # Core analysis functions and visualization tools
│   ├── search.py                      # Data collection and search functionality
│   ├── network_analysis.py            # Network analysis and relationship mapping
│   └── labeling.py                    # Classification and labeling functions
└── .venv/                             # Virtual environment (created during setup)
```

## Key Components

- **Liquid_Cooling_Market_Watch.ipynb**: Main analysis notebook containing the liquid cooling ecosystem analysis workflow
- **src/tool.py**: Core Python module with analysis functions and visualization tools
- **src/search.py**: Data collection and search functionality for liquid cooling related data
- **src/network_analysis.py**: Network analysis and relationship mapping capabilities
- **src/labeling.py**: Classification and labeling functions for provider vs. adopter identification

## Analysis Features

The liquid cooling market analysis provides:
- **Provider vs. Adopter Identification**: Automatically categorizes companies based on their role in the liquid cooling value chain
- **Technology Ecosystem Mapping**: Reveals relationships between solution providers and data center operators
- **Adoption Timeline Tracking**: Shows temporal evolution of liquid cooling implementation across sectors
- **Market Positioning Analysis**: Quantifies company exposure and positioning in the ecosystem

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook with interactive visualizations
- The tool includes async compatibility setup for Jupyter environments
- Large datasets may require significant processing time 