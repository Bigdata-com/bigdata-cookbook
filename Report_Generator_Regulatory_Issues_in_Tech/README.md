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

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

1. **Clone and navigate to the project**:
   ```bash
   cd "Report_Generator_Regulatory_Issues_in_Tech"
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
   docker build -t regulatory-issues .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 regulatory-issues
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `Report Generator_ Regulatory Issues.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t regulatory-issues .

# Run the container with port mapping
docker run -p 8888:8888 regulatory-issues

# Run in background
docker run -d -p 8888:8888 --name regulatory-issues-container regulatory-issues

# Stop the container
docker stop regulatory-issues-container

# Remove the container
docker rm regulatory-issues-container

# View logs
docker logs regulatory-issues-container

# Access container shell
docker exec -it regulatory-issues-container bash

# Remove the image
docker rmi regulatory-issues
```

## Project Structure

```