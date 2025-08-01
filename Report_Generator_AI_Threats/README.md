# AI Disruption Risk Report Generator

## Automated Analysis of AI Threats and Opportunities in Technology Companies

This project systematically evaluates both AI disruption risks and proactive AI adoption across company watchlists using unstructured data from multiple sources. Built for portfolio managers and financial analysts, it transforms scattered AI-related information into quantifiable positioning intelligence.

## Features

- **Risk-proactivity assessment** measuring both AI disruption vulnerability and strategic AI adoption initiatives
- **Standardized scoring system** enabling cross-company comparison of AI positioning and competitive readiness
- **Investment intelligence generation** revealing underlying narratives that shape each company's AI transformation journey
- **Structured output for reporting** ranking companies by AI resilience and strategic positioning

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Report_Generator_AI_Threats"
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
   docker build -t ai-threats-report-generator .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app ai-threats-report-generator
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Report Generator_ AI Disruption Risk.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to generate the AI disruption risk report

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
   cd "Report_Generator_AI_Threats"
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
   - Open `Report Generator_ AI Disruption Risk.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to generate the AI disruption risk report

## Project Structure

```
Report_Generator_AI_Threats/
├── README.md                                    # Project documentation
├── Report Generator_ AI Disruption Risk.ipynb   # Main Jupyter notebook
├── Report Generator_ AI Disruption Risk.html    # Exported HTML version
├── requirements.txt                             # Python dependencies
├── .env.example                                # Example environment variables
├── src/
│   ├── tool.py                                 # Main analysis tool
│   ├── report_generator.py                     # Report generation logic
│   ├── html_report.py                          # HTML report formatting
│   └── summary/
│       ├── summary.py                          # Summary generation
│       ├── summary_prompts.py                  # AI prompts for summaries
│       └── token_manager.py                    # Token management utilities
└── .venv/                                      # Virtual environment (created during setup)
```

## Key Components

- **Report Generator_ AI Disruption Risk.ipynb**: Main analysis notebook for AI disruption risk assessment
- **src/tool.py**: Core analysis functionality for AI threat evaluation
- **src/report_generator.py**: Automated report generation and formatting
- **src/html_report.py**: HTML report creation and styling
- **src/summary/**: AI-powered summary generation modules

## Analysis Features

The AI disruption risk analysis provides:
- **Risk Assessment**: Evaluates AI disruption vulnerability across companies
- **Proactivity Scoring**: Measures strategic AI adoption initiatives
- **Comparative Analysis**: Enables cross-company AI positioning comparison
- **Structured Reporting**: Generates ranked lists by AI resilience and positioning

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis generates both inline results and structured HTML reports
- Custom company watchlists can be modified in the notebook configuration