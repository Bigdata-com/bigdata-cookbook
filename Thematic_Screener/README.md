# Thematic Screener

## Automated Thematic Analysis and Screening Tool

This project provides comprehensive thematic analysis and screening capabilities for investment research. It's designed for portfolio managers, research analysts, and investment professionals to systematically identify, analyze, and track investment themes across various sectors and markets.

## Features

- **Thematic identification** and categorization across multiple sectors
- **Automated screening** based on thematic criteria
- **Theme tracking** and evolution analysis
- **Investment opportunity identification** through thematic lenses

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
   cd "Thematic_Screener"
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
   - Open `ThematicScreener.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

1. **Clone and navigate to the project**:
   ```bash
   cd "Thematic_Screener"
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

3. **Build the Docker image**:
   ```bash
   docker build -t thematic-screener .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 thematic-screener
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `ThematicScreener.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t thematic-screener .

# Run the container with port mapping
docker run -p 8888:8888 thematic-screener

# Run in background
docker run -d -p 8888:8888 --name thematic-screener-container thematic-screener

# Stop the container
docker stop thematic-screener-container

# Remove the container
docker rm thematic-screener-container

# View logs
docker logs thematic-screener-container

# Access container shell
docker exec -it thematic-screener-container bash

# Remove the image
docker rmi thematic-screener
```

## Project Structure

```
Thematic_Screener/
├── README.md                 # Project documentation
├── ThematicScreener.ipynb    # Main Jupyter notebook for thematic analysis
├── ThematicScreener.html     # Exported HTML version of the notebook
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker configuration
├── .dockerignore            # Docker ignore file
├── .env.example             # Example environment variables
├── src/
│   └── tool.py              # Core thematic screening functionality
└── .venv/                   # Virtual environment (created during setup)
```

## Key Components

- **ThematicScreener.ipynb**: Main analysis notebook containing the thematic screening workflow
- **src/tool.py**: Core Python module with thematic analysis and screening functions

## Analysis Features

The thematic screener provides:
- **Theme Identification**: Automated detection and categorization of investment themes
- **Sector Coverage**: Comprehensive analysis across multiple market sectors
- **Screening Capabilities**: Filtering based on thematic criteria and thresholds
- **Opportunity Detection**: Investment opportunity identification through thematic lenses

## Thematic Categories Covered

- **Technology Themes**: AI/ML, Cloud Computing, Cybersecurity, Fintech
- **Sustainability Themes**: Clean Energy, ESG, Carbon Reduction, Green Tech
- **Healthcare Themes**: Biotech, Digital Health, Precision Medicine, Telemedicine
- **Consumer Themes**: E-commerce, Digital Payments, Streaming, Gaming
- **Industrial Themes**: Automation, IoT, Smart Manufacturing, Supply Chain Tech

## Usage Notes

- Ensure all credentials are properly configured in the `.env` file before running
- The notebook should be run sequentially from top to bottom
- Analysis results are displayed inline in the notebook
- Custom thematic criteria can be modified in the notebook configuration
- Graphviz installation is required for visualization features