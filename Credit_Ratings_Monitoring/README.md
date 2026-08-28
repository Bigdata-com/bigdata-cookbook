# Credit Ratings Monitoring

## Automated Detection and Analysis of Credit Rating Events

This project provides comprehensive credit rating monitoring capabilities for financial analysts, portfolio managers, and credit risk professionals. It systematically detects, labels, and summarizes credit rating-related news events using the Bigdata API and large language models to transform unstructured content into structured insights.

## Features

- **Event Detection & Classification** - Automated identification of credit rating updates, outlook changes, and watch list events
- **Entity Relationship Mapping** - Distinguishes between rating agencies (raters) and rated entities (ratees) with validation workflows
- **Multi-Feature Extraction** - Captures credit ratings, outlooks, watchlist status, debt instruments, and key drivers
- **Timeline Analysis** - Generates chronological reports showing rating evolution over time
- **Interactive Visualizations** - Creates HTML reports with interactive charts for rating timeline analysis

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Credit_Ratings_Monitoring"
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
   docker build -t credit-ratings-monitoring .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8888:8888 --env-file .env -v "$(pwd)":/app credit-ratings-monitoring
   ```

4. **Access JupyterLab**:
   - Open your browser and navigate to `http://localhost:8888`
   - Open `Credit_Ratings_Monitoring.ipynb`
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
   cd "Credit_Ratings_Monitoring"
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
   - Open `Credit_Ratings_Monitoring.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Credit_Ratings_Monitoring/
├── README.md                           # Project documentation
├── Credit_Ratings_Monitoring.ipynb     # Main Jupyter notebook for credit ratings analysis
├── Credit_Ratings_Monitoring.html      # Exported HTML version of the notebook
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker container configuration
├── .dockerignore                       # Docker ignore patterns
├── .env.example                       # Example environment variables
├── src/
│   ├── feature_extractor.py           # Core feature extraction and LLM processing
│   ├── knowledge_graph_manager.py     # Entity ID retrieval and management
│   ├── search_enhanced.py             # Enhanced search with context retrieval
│   ├── summary_generator.py           # Report generation and summarization
│   └── visuals.py                     # Visualization and HTML report generation
├── report/                            # Generated analysis reports
│   ├── credit_ratings_monitor.csv
│   ├── tesla_inc_credit_ratings_report.html
│   └── tesla_credit_ratings_timeline.html
└── .venv/                             # Virtual environment (created during setup)
```

## Key Components

- **Credit_Ratings_Monitoring.ipynb**: Main analysis notebook containing the complete credit rating monitoring workflow
- **src/feature_extractor.py**: LLM-powered feature extraction for credit ratings, outlooks, and related metadata
- **src/knowledge_graph_manager.py**: Entity resolution and relationship mapping using Bigdata's knowledge graph
- **src/search_enhanced.py**: Enhanced search functionality with contextual chunk retrieval
- **src/summary_generator.py**: Timeline generation and event summarization capabilities
- **src/visuals.py**: Interactive visualization and HTML report generation

## Analysis Features

The credit ratings monitoring provides:

### Content Retrieval & Enhancement
- **Parallel Search Processing**: Configurable time windows with batch processing for efficient data collection
- **Contextual Enhancement**: Retrieval of surrounding content paragraphs for richer analysis context
- **Entity-Keyword Filtering**: Focused search combining target companies, rating agencies, and credit-related keywords

### Feature Extraction & Validation
- **Entity Role Detection**: Distinguishes between rating agencies (raters) and rated entities (ratees)
- **Multi-Step Validation**: LLM-powered validation workflow to confirm and correct entity role assignments
- **Structured Feature Extraction**: Captures multiple credit-related features across three specialized prompts:

#### Credit Rating Features
- **Credit Rating**: Overall rating assignments and changes
- **Credit Action**: Upgrades, downgrades, affirmations, corrections, withdrawals, and reinstatements
- **Credit Status**: Provisional ratings, maturity events, and publication status
- **Credit Outlook**: Positive, negative, stable, or developing outlook assessments
- **Credit Watchlist**: Watch placement, removal, and directional indicators

#### Debt Instrument Analysis
- **Short-Term Ratings**: Commercial paper and short-term debt instrument ratings
- **Long-Term Ratings**: Bond and long-term debt instrument ratings
- **Instrument Classification**: Detailed categorization of rated debt instruments

#### Key Drivers & Guidance
- **Rating Drivers**: Cash flow, earnings, capital structure, and operational factors influencing ratings
- **Forward Guidance**: Future rating expectations and outlook commentary
- **Market Context**: Related financial metrics and market condition discussions

### Advanced Analytics Generation
- **Timestamped Datasets**: Structured data with precise temporal attribution for trend analysis
- **Entity Relationship Tracking**: Monitoring of rater-ratee relationships over time
- **Confidence Scoring**: Quality assessment for extracted features and relationships

### Report Generation
- **Timeline Visualization**: Interactive charts showing rating evolution over time
- **Event Summarization**: Daily summaries removing duplicates and highlighting new information
- **Structured Reporting**: Exportable datasets for further analysis and integration
- **HTML Dashboards**: Professional reports with supporting quotes, source links, and metadata

## Credit Rating Events Covered

- **Rating Actions**: Upgrades, downgrades, affirmations, withdrawals, and reinstatements
- **Outlook Changes**: Positive, negative, stable, and developing outlook revisions
- **Watchlist Events**: Watch placements, removals, and directional watch indicators
- **Coverage Decisions**: New coverage initiations and coverage terminations
- **Methodology Updates**: Rating criteria changes and methodological revisions

## Supported Rating Agencies

The framework is designed to work with major credit rating agencies including:
- **S&P Global Ratings**
- **Moody's Investors Service**
- **Fitch Ratings**
- **Regional and Specialized Agencies**: Configurable for additional rating organizations

## Real-World Use Case

The project demonstrates the complete workflow through tracking credit rating updates for Tesla over a three-year period, showing how to:
- Transform unstructured rating news into structured insights
- Identify rater-ratee relationships with validation
- Extract analyst commentary and market implications
- Generate timeline reports with supporting documentation
- Create exportable datasets for quantitative analysis

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are automatically saved to the `report/` directory
- Large date ranges may require significant processing time due to LLM feature extraction
- Custom rating criteria and entities can be modified in the notebook configuration
- Interactive visualizations require modern browser with JavaScript enabled
