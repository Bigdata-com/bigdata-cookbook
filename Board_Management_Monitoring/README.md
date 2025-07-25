# Board Management Monitoring

## Automated Analysis of Board Member and Management Activity Exposure

This project provides comprehensive board management monitoring capabilities for investment research and governance analysis. It's designed for analysts, portfolio managers, and investment professionals to systematically track specific individuals across news coverage, providing insights into management activity and board dynamics.

## Features

- **Comprehensive person tracking** across multiple name variations and contexts
- **Company-specific filtering** ensuring relevance to the monitored organization
- **Multi-mode search precision** from strict entity matching to broader coverage with post-filtering
- **Source filtering** enabling focused analysis across trusted news sources
- **Temporal analysis** showing how coverage patterns evolve over time
- **Entity-specific monitoring** using bigdata's entity tracking capabilities

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
   cd "Board_Management_Monitoring"
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   uv pip install jupyterlab
   ```

4. **Set up credentials**:
   - Create a `.env` file in the project directory:
     ```bash
     touch .env
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
   - Open `Board_Management_Monitoring.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Project Structure

```
Board_Management_Monitoring/
├── README.md                           # Project documentation
├── Board_Management_Monitoring.ipynb   # Main Jupyter notebook for board monitoring
├── Board_Management_Monitoring.html    # Exported HTML version of the notebook
├── requirements.txt                    # Python dependencies
├── src/
│   └── tool.py                        # Core board monitoring functionality
└── .venv/                             # Virtual environment (created during setup)
```

## Key Components

- **Board_Management_Monitoring.ipynb**: Main analysis notebook containing the board monitoring workflow
- **src/tool.py**: Core Python module with entity tracking and monitoring functions

## Analysis Features

The board management monitoring provides:
- **Entity Tracking**: Systematic monitoring of specific individuals across news coverage
- **Multi-Mode Search**: Strict entity matching to broader coverage with post-filtering
- **Company Filtering**: Ensures relevance to the monitored organization
- **Temporal Analysis**: Shows how coverage patterns evolve over time
- **Source Filtering**: Focused analysis across trusted news sources

## Search Modes

- **Strict Mode**: Precise entity matching for high-confidence results
- **Relaxed Mode**: Broader coverage with post-filtering for comprehensive analysis
- **Relaxed Post Mode**: Extended coverage with additional filtering criteria

## Monitoring Capabilities

- **Person Tracking**: Monitor specific individuals across multiple name variations
- **Management Themes**: Track mentions related to management integrity and governance
- **Board Governance**: Analyze board-related activities and decisions
- **Media Exposure**: Quantify and analyze media coverage patterns
- **Risk Assessment**: Identify potential governance and management risks

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Customize person configurations and company settings in the notebook
- Analysis results are displayed inline with interactive visualizations
- Large datasets may require significant processing time

## Real-World Applications

This workflow is particularly useful for:
- **Governance Analysis**: Monitoring board member activities and decisions
- **Management Tracking**: Following executive mentions and activities
- **Risk Assessment**: Identifying potential governance or management issues
- **Investment Research**: Understanding management dynamics and reputation signals
- **Compliance Monitoring**: Tracking regulatory and compliance-related mentions 