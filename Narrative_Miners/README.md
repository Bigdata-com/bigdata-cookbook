# NarrativeMiner

## Automated Narrative Analysis and Mining Tool

This project provides comprehensive narrative analysis and mining capabilities for financial and investment research. It's designed for analysts, portfolio managers, and investment professionals to systematically extract, analyze, and track narrative patterns across various data sources and market segments.

## Features

- **Narrative extraction** and pattern recognition from unstructured data
- **Sentiment analysis** and narrative sentiment tracking
- **Narrative evolution** and temporal analysis
- **Automated narrative scoring** and ranking systems

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
   cd "Narrative_Miners"
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
   - Open `NarrativeMiner.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

1. **Clone and navigate to the project**:
   ```bash
   cd "Narrative_Miners"
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
   docker build -t narrative-miner .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 narrative-miner
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `NarrativeMiner.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t narrative-miner .

# Run the container with port mapping
docker run -p 8888:8888 narrative-miner

# Run in background
docker run -d -p 8888:8888 --name narrative-miner-container narrative-miner

# Stop the container
docker stop narrative-miner-container

# Remove the container
docker rm narrative-miner-container

# View logs
docker logs narrative-miner-container

# Access container shell
docker exec -it narrative-miner-container bash

# Remove the image
docker rmi narrative-miner
```

## Project Structure

```