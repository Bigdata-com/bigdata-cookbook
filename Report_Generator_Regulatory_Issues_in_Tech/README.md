# Regulatory Issues in Tech Report Generator

## Automated Analysis of Regulatory Risks and Company Mitigation Strategies

This project systematically analyzes regulatory exposure across company watchlists using unstructured data from news, filings, and transcripts. Built for risk managers and investment professionals, it transforms scattered regulatory information into quantifiable risk intelligence.

## Features

- **Sector-wide regulatory mapping** across technology domains (AI, Social Media, Hardware & Chips, E-commerce, Advertising)
- **Company-specific risk quantification** using Media Attention, Risk/Financial Impact, and Uncertainty metrics
- **Mitigation strategy extraction** from corporate communications to identify compliance approaches
- **Structured output for reporting** ranking regulatory issues by intensity and business impact

## Local Installation and Usage

### Prerequisites
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Graphviz](https://pypi.org/project/graphviz/) - Required for graph visualization features

### Setup and Run

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
   cd "Report_Generator_Regulatory_Issues_in_Tech"
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
   - Open `Report Generator_ Regulatory Issues.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to generate the regulatory risk report

## Project Structure

```
Report_Generator_Regulatory_Issues_in_Tech/
├── README.md                                    # Project documentation
├── Report Generator_ Regulatory Issues.ipynb    # Main Jupyter notebook
├── Report Generator_ Regulatory Issues.html     # Exported HTML version
├── requirements.txt                             # Python dependencies
├── .env.example                                # Example environment variables
├── src/
│   ├── tool.py                                 # Main analysis tool
│   ├── report_generator.py                     # Report generation logic
│   ├── html_report.py                          # HTML report formatting
│   ├── summary/
│   │   ├── summary.py                          # Summary generation
│   │   ├── summary_prompts.py                  # AI prompts for summaries
│   │   └── token_manager.py                    # Token management utilities
│   └── response/
│       ├── company_response.py                 # Company response analysis
│       └── response_prompts.py                 # Response analysis prompts
└── .venv/                                      # Virtual environment (created during setup)
```

## Key Components

- **Report Generator_ Regulatory Issues.ipynb**: Main analysis notebook for regulatory risk assessment
- **src/tool.py**: Core analysis functionality for regulatory issue evaluation
- **src/report_generator.py**: Automated report generation and formatting
- **src/html_report.py**: HTML report creation and styling
- **src/summary/**: AI-powered summary generation modules
- **src/response/**: Company response and mitigation strategy analysis

## Analysis Features

The regulatory issues analysis provides:
- **Sector Mapping**: Comprehensive regulatory coverage across technology domains
- **Risk Quantification**: Media attention, financial impact, and uncertainty metrics
- **Mitigation Analysis**: Company response and compliance strategy extraction
- **Structured Reporting**: Ranked regulatory issues by intensity and business impact

## Technology Domains Covered

- **AI & Machine Learning**: AI regulation, data privacy, algorithmic bias
- **Social Media**: Content moderation, user privacy, platform responsibility
- **Hardware & Chips**: Export controls, supply chain security, intellectual property
- **E-commerce**: Antitrust, consumer protection, data security
- **Advertising**: Privacy regulations, targeting restrictions, transparency

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis generates both inline results and structured HTML reports
- Custom company watchlists can be modified in the notebook configuration
- Graphviz installation is required for visualization features