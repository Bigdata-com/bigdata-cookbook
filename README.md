# Bigdata Cookbook

A comprehensive collection of financial analysis tools and report generators built on the Bigdata API and research tools. This repository contains ready-to-use notebooks for thematic screening, narrative mining, and various sector-specific analyses including pricing power, AI disruption risks, and regulatory issues in the technology sector.

## Projects

### 🔍 [Thematic Screener](./Thematic_Screener/)
**Automated Thematic Analysis and Screening Tool**

- Thematic identification and categorization across multiple sectors
- Automated screening based on thematic criteria
- Theme tracking and evolution analysis
- Investment opportunity identification through thematic lenses

### 📊 [Pricing Power Analysis](./Pricing_Power_Analysis/)
**Automated Analysis of Pricing Power Narratives and Competitive Positioning**

- Assesses competitive positioning across company watchlists
- Provides sector-wide comparative analysis
- Tracks temporal evolution of pricing narratives
- Implements confidence scoring system for pricing power signals

### 🤖 [AI Threats Report Generator](./Report_Generator_AI_Threats/)
**Automated Analysis of AI Threats and Opportunities in Technology Companies**

- Evaluates AI disruption risks and proactive AI adoption
- Provides standardized scoring for cross-company comparison
- Generates investment intelligence from AI transformation narratives
- Creates structured reports ranking companies by AI resilience

### ⚖️ [Regulatory Issues in Tech Report Generator](./Report_Generator_Regulatory_Isses_in_Tech/)
**Automated Analysis of Regulatory Risks and Company Mitigation Strategies**

- Maps sector-wide regulatory issues across technology domains
- Quantifies company-specific regulatory risks
- Extracts mitigation strategies from corporate communications
- Provides structured reporting on regulatory intensity and business impact

### 🎯 [Risk Analyzer](./Risk_Analyzer/)
**Automated Risk Analysis and Assessment Tool**

- Comprehensive risk assessment across multiple risk dimensions
- Quantitative risk modeling with statistical analysis
- Risk visualization and reporting capabilities
- Automated risk scoring and ranking systems

### 📖 [Narrative Miners](./Narrative_Miners/)
**Automated Narrative Analysis and Mining Tool**

- Narrative extraction and pattern recognition from unstructured data
- Sentiment analysis and narrative sentiment tracking
- Narrative evolution and temporal analysis
- Automated narrative scoring and ranking systems

### 👥 [Board Management Monitoring](./Board_Management_Monitoring/)
**Automated Analysis of Board Member and Management Activity Exposure**

- Comprehensive person tracking across multiple name variations and contexts
- Company-specific filtering ensuring relevance to monitored organizations
- Multi-mode search precision from strict entity matching to broader coverage
- Temporal analysis showing how coverage patterns evolve over time
- Entity-specific monitoring using bigdata's entity tracking capabilities

## Quick Start

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Bigdata API access
- OpenAI API key (for advanced features)

### Installation

Each project has its own detailed README with specific installation and usage instructions.

## Project Structure

```
bigdata-cookbook/
├── Pricing_Power_Analysis/                          # Pricing power analysis
│   ├── Pricing Power.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_AI_Threats/                      # AI risk analysis
│   ├── Report Generator_ AI Disruption Risk.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_Regulatory_Isses_in_Tech/        # Regulatory analysis
│   ├── Report Generator_ Regulatory Issues.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Risk_Analyzer/                                    # Risk analysis tool
│   ├── Risk_Analyzer.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Thematic_Screener/                                # Thematic analysis tool
│   ├── ThematicScreener.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Narrative_Miners/                                 # Narrative analysis tool
│   ├── NarrativeMiner.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Board_Management_Monitoring/                      # Board monitoring tool
│   ├── Board_Management_Monitoring.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
└── README.md                                        # This file
```

## Features

- **Client-Ready**: Each project is self-contained with its own dependencies and documentation
- **Easy Setup**: Uses uv for fast, reliable dependency management
- **Comprehensive Analysis**: Combines multiple data sources for robust insights
- **Professional Output**: Generates Excel reports, HTML visualizations, and structured data
- **Modular Design**: Each project can be run independently

## Requirements

### Core Dependencies
- `bigdata-client>=2.17.0` - Bigdata API client
- `bigdata-research-tools==0.17.2` - Research analysis tools
- `nest-asyncio>=1.6.0` - Async compatibility
- `matplotlib>=3.0.0` - Data visualization
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `jupyter>=1.0.0` - Notebook environment

### Optional Dependencies
- `seaborn>=0.11.0` - Statistical visualizations
- `plotly>=5.0.0` - Interactive plots
- `ipython>=7.0.0` - Enhanced Python shell

## Usage

Each project follows a similar workflow:

1. **Setup**: Install dependencies and configure credentials
2. **Data Collection**: Fetch relevant data from Bigdata platform
3. **Analysis**: Run the analysis pipeline
4. **Reporting**: Generate Excel and HTML reports
5. **Visualization**: Create charts and insights

## Support

- Each project has its own detailed README with specific instructions
- Check the individual project documentation for troubleshooting
- Ensure you have valid Bigdata API credentials before running analyses

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This repository contains financial analysis tools. Please ensure compliance with relevant regulations and use appropriate risk management practices when making investment decisions based on these analyses.
