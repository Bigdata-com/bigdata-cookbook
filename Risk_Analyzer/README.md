# Risk Analyzer

## Automated Risk Analysis and Assessment Tool

This project provides comprehensive risk analysis capabilities for financial and investment analysis. It's designed for risk managers, portfolio managers, and investment professionals to systematically assess and quantify various types of risks across different asset classes and investment strategies.

## Features

- **Comprehensive risk assessment** across multiple risk dimensions
- **Quantitative risk modeling** with statistical analysis
- **Risk visualization** and reporting capabilities
- **Automated risk scoring** and ranking systems

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Risk_Analyzer"
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
   docker build -t risk-analyzer .
   
   # Run the container
   docker run -p 8888:8888 --env-file .env -v $(pwd):/app risk-analyzer
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Risk_Analyzer.ipynb`
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
   cd "Risk_Analyzer"
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
   - Open `Risk_Analyzer.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Risk_Analyzer/
├── README.md                 # Project documentation
├── Risk_Analyzer.ipynb       # Main Jupyter notebook for risk analysis
├── Risk_Analyzer.html        # Exported HTML version of the notebook
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── src/
│   └── tool.py              # Core risk analysis functionality
├── risk-analyzer/           # Additional risk analysis modules
└── .venv/                   # Virtual environment (created during setup)
```

## Key Components

- **Risk_Analyzer.ipynb**: Main analysis notebook containing the risk assessment workflow
- **src/tool.py**: Core Python module with risk analysis and modeling functions
- **risk-analyzer/**: Additional risk analysis modules and utilities

## Analysis Features

The risk analyzer provides:
- **Multi-dimensional Risk Assessment**: Evaluates various risk types across different dimensions
- **Quantitative Modeling**: Statistical analysis and risk quantification
- **Visualization**: Risk charts, graphs, and reporting capabilities
- **Automated Scoring**: Risk ranking and scoring systems

## Risk Dimensions Covered

- **Market Risk**: Price volatility, correlation analysis, beta calculations
- **Credit Risk**: Default probability, credit spreads, rating analysis
- **Operational Risk**: Business continuity, process failures, system risks
- **Liquidity Risk**: Trading volume, bid-ask spreads, market depth
- **Regulatory Risk**: Compliance requirements, policy changes, legal exposure

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook
- Custom risk parameters can be modified in the notebook configuration
- Graphviz installation is required for visualization features