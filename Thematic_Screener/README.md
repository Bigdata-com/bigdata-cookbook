# Thematic Screener

> **Preferred approach:** Use [`Thematic_Screener_CLI`](../Thematic_Screener_CLI/) for production workflows (CLI, MCP, derivative modes, full artifact export). This folder contains a simplified Jupyter walkthrough that delegates to the same REST pipeline. See [Thematic Screeners](https://docs.bigdata.com/use-cases/research-tools/screeners) on docs.bigdata.com.

## Automated Thematic Analysis and Screening Tool

This project provides comprehensive thematic analysis and screening capabilities for investment research. It's designed for portfolio managers, research analysts, and investment professionals to systematically identify, analyze, and track investment themes across various sectors and markets.

## Features

- **Thematic identification** and categorization across multiple sectors
- **Automated screening** based on thematic criteria
- **Theme tracking** and evolution analysis
- **Investment opportunity identification** through thematic lenses

## Prerequisites

- A clone of this cookbook repo with both `Thematic_Screener/` and `Thematic_Screener_CLI/` present
- [Bigdata.com API key](https://docs.bigdata.com/api-reference/authentication) and OpenAI API key

## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone the cookbook and navigate to the repo root**:
   ```bash
   cd bigdata-cookbook
   ```

2. **Set up credentials** in `Thematic_Screener/`:
   ```bash
   cp Thematic_Screener/.env.example Thematic_Screener/.env
   ```
   Edit `Thematic_Screener/.env`:
   ```
   BIGDATA_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Build and run the Docker container** (build context is the cookbook root):
   ```bash
   docker build -f Thematic_Screener/Dockerfile -t thematic-screener .
   docker run -u "$(id -u):$(id -g)" -e HOME=/cookbook \
     -p 8888:8888 --env-file Thematic_Screener/.env \
     -v "$(pwd)":/cookbook thematic-screener
   ```

4. **Access JupyterLab**:
   - Open `http://localhost:8888`
   - Open `ThematicScreener.ipynb`
   - Run cells sequentially

### Option 2: Local Installation

#### Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

#### Setup and Run

1. **Install dependencies from the CLI package** (recommended):
   ```bash
   cd Thematic_Screener_CLI
   uv sync --group jupyter
   ```

2. **Set up credentials**:
   ```bash
   cp .env.example .env   # in Thematic_Screener_CLI/ or Thematic_Screener/
   ```
   Edit `.env`:
   ```
   BIGDATA_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Start JupyterLab** from the CLI directory:
   ```bash
   uv run jupyter lab ../Thematic_Screener/ThematicScreener.ipynb
   ```

   Alternatively, from `Thematic_Screener/`:
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   jupyter lab
   ```

4. **Run the notebook** sequentially from top to bottom.

## Project Structure

```
Thematic_Screener/
├── README.md                 # This file
├── ThematicScreener.ipynb    # Walkthrough notebook (delegates to CLI pipeline)
├── ThematicScreener.html     # Exported HTML (may lag behind notebook)
├── requirements.txt          # Points to Thematic_Screener_CLI deps + Jupyter
├── Dockerfile                # Builds from cookbook root
├── .env.example              # API key template
└── src/                      # Legacy folder (no SDK code)
```

## Key Components

- **ThematicScreener.ipynb**: Orchestrates the four-stage CLI pipeline for a Supply Chain Reshaping screen over XNAS top 100
- **[Thematic_Screener_CLI](../Thematic_Screener_CLI/)**: Preferred implementation — CLI, MCP, derivative modes, and full run artefacts

## Usage Notes

- Ensure `BIGDATA_API_KEY` and `OPENAI_API_KEY` are set before running
- The notebook writes artefacts to `Thematic_Screener_CLI/runs/xnas_supply_chain_exposure/`
- For derivative screens and MCP workflows, use `Thematic_Screener_CLI` directly
