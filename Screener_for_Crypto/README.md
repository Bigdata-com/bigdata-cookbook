# Screener for Crypto

A comprehensive thematic screening tool for cryptocurrency analysis using the Bigdata platform. This tool systematically identifies and quantifies cryptocurrency exposure to specific investment themes through automated content analysis and LLM-powered classification.

## 🚀 Features

- **Automated Theme Taxonomy Generation**: Uses LLM to break down complex themes into measurable sub-categories
- **Cryptocurrency Screening**: Analyzes top cryptocurrencies against specific investment themes
- **Semantic Content Retrieval**: Leverages Bigdata's search capabilities to find relevant content
- **LLM-Powered Classification**: Uses advanced language models to classify and score thematic relevance
- **Interactive Visualizations**: Creates comprehensive charts and heatmaps for analysis

## 📊 What It Does

The Screener for Crypto tool helps investors and analysts:

- **Identify Thematic Exposure**: Quantify how different cryptocurrencies align with specific investment themes
- **Cross-Crypto Comparison**: Compare thematic positioning across multiple cryptocurrencies
- **Investment Intelligence**: Transform narrative signals into structured, actionable insights

## 🎯 Use Cases

- **Thematic Investing**: Identify cryptocurrencies positioned to benefit from specific trends
- **Portfolio Construction**: Build crypto portfolios based on thematic exposure
- **Risk Assessment**: Understand thematic concentration risks in crypto holdings
- **Market Research**: Analyze how different cryptocurrencies are positioned in emerging trends

## 🛠️ Installation

### Prerequisites

- Docker installed on your system
- Bigdata API access credentials
- OpenAI API key (for LLM-powered analysis)

### Docker Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bigdata-com/bigdata-cookbook.git
   cd bigdata-cookbook/Screener_for_Crypto
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```env
   BIGDATA_USERNAME=your_bigdata_username
   BIGDATA_PASSWORD=your_bigdata_password
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Build and run with Docker**:
   ```bash
   # Build the Docker image
   docker build -t crypto-screener .
   
   # Run the container
   docker run -p 8888:8888 --env-file .env -v $(pwd)/output:/app/output crypto-screener
   ```

4. **Access Jupyter Notebook**:
   - Open your browser and go to `http://localhost:8888`
   - Open `Screener_for_Crypto.ipynb`

### Local Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   export BIGDATA_USERNAME="your_username"
   export BIGDATA_PASSWORD="your_password"
   export OPENAI_API_KEY="your_api_key"
   ```

3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook Screener_for_Crypto.ipynb
   ```

## 📖 Usage

### Basic Workflow

1. **Define Your Theme**: Set the main investment theme you want to analyze
2. **Configure Parameters**: Set date ranges, document types, and analysis parameters
3. **Run Analysis**: Execute the screening pipeline
4. **Review Results**: Analyze the generated visualizations and reports

### Example Configuration

```python
# Define your theme
main_theme = "Crypto Cross-Chain Interoperability"
focus = ""

# Set analysis parameters
start_date = "2025-01-01"
end_date = "2025-09-08"
document_type = DocumentType.NEWS
sources = ["D6D057"]  # Crypto Wire source
```

### Key Parameters

- **Main Theme**: The central investment concept to explore
- **Entity Universe**: Provide as:
  - CSV file with columns `RP_ENTITY_ID` (or `RP_COMPANY_ID`) and `COMPANY_NAME` (or `NAME`), OR
  - Python list/dict of entity IDs (e.g., `["4F2B", "D8442"]`) with names
- **Time Period**: Date range for analysis
- **Sources**: Specific news sources to include (optional)
- **LLM Model**: OpenAI model for theme generation and classification

## 📁 Project Structure

```
Screener_for_Crypto/
├── Screener_for_Crypto.ipynb    # Main analysis notebook
├── requirements.txt              # Python dependencies
├── Dockerfile                   # Docker configuration
├── .dockerignore               # Docker ignore file
├── README.md                   # This file
├── src/                        # Source code modules
│   ├── search_entities.py      # Entity search functionality
│   └── visualization_tool.py   # Visualization utilities
└── output/                     # Generated reports and visualizations
    └── thematic_screener_results.xlsx
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `BIGDATA_API_KEY` | Your Bigdata.com API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key for LLM analysis | Yes |

### Analysis Parameters

- **Theme Definition**: Customize the investment theme and focus area
- **Entity Selection**: Pass a CSV of crypto entity IDs (`RP_ENTITY_ID`, `COMPANY_NAME`) or an explicit ID list
- **Time Range**: Set analysis period (supports daily, weekly, monthly, yearly intervals)
- **Document Sources**: Filter by specific news sources or document types
- **LLM Settings**: Configure model selection and classification parameters

## 📊 Output

The tool generates several types of output:

### 1. Excel Report
- **File**: `output/thematic_screener_results.xlsx`
- **Content**: Detailed results with entity scores, themes, and supporting quotes
- **Columns**: Time Period, Date, Entity, Document ID, Headline, Quote, Motivation, Theme

### 2. Visualizations
- **Entity Thematic Exposure Heatmap**: Shows raw scores for each crypto across themes
- **Composite Score Bar Chart**: Ranks cryptocurrencies by total thematic exposure
- **Top Themes Scatter Plot**: Displays top 3 thematic exposures for each entity
- **Theme Summary Chart**: Shows total scores across all themes

### 3. HTML Reports
- Interactive visualizations embedded in HTML format
- Exportable charts and graphs
- Professional presentation-ready outputs

## 🎨 Visualization Features

### Interactive Charts
- **Plotly Integration**: Interactive, zoomable, and exportable charts
- **Multiple Chart Types**: Heatmaps, bar charts, scatter plots, and summary visualizations
- **Responsive Design**: Adapts to different screen sizes and devices

### Static Charts
- **Matplotlib Support**: High-quality static images for reports
- **Custom Styling**: Professional color schemes and formatting
- **Export Options**: PNG, PDF, and other image formats

## 🔍 Advanced Features

### Theme Taxonomy Generation
- **LLM-Powered**: Uses advanced language models to break down complex themes
- **Hierarchical Structure**: Creates organized theme trees with sub-categories
- **Contextual Relevance**: Ensures all sub-themes connect back to the main theme

### Semantic Search
- **Bigdata Integration**: Leverages Bigdata's advanced search capabilities
- **Entity Recognition**: Automatically identifies and tracks cryptocurrency mentions
- **Content Filtering**: Filters results for thematic relevance

### Classification Pipeline
- **LLM Classification**: Uses language models to score thematic relevance
- **Confidence Scoring**: Provides confidence levels for classifications
- **Quality Control**: Filters out unclear or irrelevant content

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify your Bigdata credentials in the `.env` file
   - Ensure your account has access to the required data sources

2. **API Rate Limits**:
   - The tool includes rate limiting and retry logic
   - Consider reducing batch sizes for large analyses

3. **Memory Issues**:
   - For large analyses, consider processing in smaller batches
   - Monitor system resources during execution

4. **Docker Issues**:
   - Ensure Docker is running and has sufficient resources
   - Check port 8888 is not already in use

### Performance Tips

- **Batch Processing**: Use appropriate batch sizes for your system
- **Date Ranges**: Start with shorter time periods for testing
- **Entity Limits**: Begin with smaller entity sets for initial analysis

## 📚 Dependencies

### Core Dependencies
- `bigdata-smart-batching>=1.3.1` - Plan/execute search over entity universes
- `requests>=2.31.0` - REST calls to Bigdata.com
- `openai>=1.0.0` - Local labeling / taxonomy
- `python-dotenv>=1.0.0` - `BIGDATA_API_KEY` loading
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.20.0` - Numerical computing

> Do not install `bigdata-client` or `bigdata-research-tools`.

### Visualization
- `plotly>=5.0.0` - Interactive visualizations
- `matplotlib>=3.0.0` - Static plotting
- `seaborn>=0.11.0` - Statistical visualizations

### Jupyter Environment
- `jupyter>=1.0.0` - Notebook environment
- `ipython>=7.0.0` - Enhanced Python shell

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🆘 Support

- Check the [Bigdata Documentation](https://sdk.bigdata.com/) for API details
- Review the main [Bigdata Cookbook README](../README.md) for general guidance
- Ensure you have valid API credentials before running analyses

---

**Note**: This tool is designed for financial analysis and research purposes. Please ensure compliance with relevant regulations and use appropriate risk management practices when making investment decisions based on these analyses.
