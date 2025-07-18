# Pricing Power Analysis

## Automated Analysis of Pricing Power Narratives and Competitive Positioning

This project provides comprehensive pricing power analysis tools that assess competitive positioning across company watchlists using unstructured data from news sources. It's designed for analysts, portfolio managers, and investment professionals to transform scattered pricing signals into quantified competitive intelligence.

## Features

- **Positive vs. negative pricing power assessment** measuring both pricing strength and competitive pressure signals
- **Sector-wide comparative analysis** revealing industry patterns and competitive positioning dynamics
- **Temporal evolution tracking** showing how pricing narratives develop and change over time
- **Confidence scoring system** quantifying the balance between positive and negative pricing power mentions

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
   cd "Pricing_Power_Analysis"
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
   - Open `Pricing Power.ipynb`
   - Follow the setup instructions in the notebook
   - Run cells sequentially to perform the analysis

## Docker Installation and Usage

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine

### Setup and Run

1. **Clone and navigate to the project**:
   ```bash
   cd "Pricing_Power_Analysis"
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
   docker build -t pricing-power .
   ```

4. **Run the container**:
   ```bash
   docker run -p 8888:8888 pricing-power
   ```

5. **Access JupyterLab**:
   - When the server starts, it will display one or more access URLs in the terminal
   - Click on one of the provided URLs (or copy and paste it into your browser) to open JupyterLab
   - Open `Pricing Power.ipynb` and start analyzing

### Docker Commands

```bash
# Build the Docker image
docker build -t pricing-power .

# Run the container with port mapping
docker run -p 8888:8888 pricing-power

# Run in background
docker run -d -p 8888:8888 --name pricing-power-container pricing-power

# Stop the container
docker stop pricing-power-container

# Remove the container
docker rm pricing-power-container

# View logs
docker logs pricing-power-container

# Access container shell
docker exec -it pricing-power-container bash

# Remove the image
docker rmi pricing-power
```

## Project Structure

```