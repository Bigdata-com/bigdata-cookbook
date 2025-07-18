# Risk Analyzer

## Automated Risk Analysis and Assessment Tool

This project provides comprehensive risk analysis capabilities for financial and investment analysis. It's designed for risk managers, portfolio managers, and investment professionals to systematically assess and quantify various types of risks across different asset classes and investment strategies.

## Features

- **Comprehensive risk assessment** across multiple risk dimensions
- **Quantitative risk modeling** with statistical analysis
- **Risk visualization** and reporting capabilities
- **Automated risk scoring** and ranking systems

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

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

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

3. **Build the Docker image**:
   ```bash
   docker build -t risk-analyzer .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 risk-analyzer
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `Risk_Analyzer.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t risk-analyzer .

# Run the container with port mapping
docker run -p 8888:8888 risk-analyzer

# Run in background
docker run -d -p 8888:8888 --name risk-analyzer-container risk-analyzer

# Stop the container
docker stop risk-analyzer-container

# Remove the container
docker rm risk-analyzer-container

# View logs
docker logs risk-analyzer-container

# Access container shell
docker exec -it risk-analyzer-container bash

# Remove the image
docker rmi risk-analyzer
```

## Project Structure

```