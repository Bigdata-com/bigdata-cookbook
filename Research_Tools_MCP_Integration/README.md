# Research Tools Integration with MCP

This project showcases how to integrate the Bigdata research tools with an MCP server, enabling the creation of watchlists and thematic screening of companies.


## Installation and Usage

### Option 1: Docker Installation

#### Prerequisites
- Docker installed on your system

#### Setup and Run with Docker

1. **Clone and navigate to the project**:
   ```bash
   cd "Research_Tools_MCP_Integration"
   ```

2. **Set up credentials**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your credentials:
     ```
     BIGDATA_API_KEY=your_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

3. **Build and run the Docker container**:
   ```bash
   # Build the Docker image
   docker build -t mcp-thematic-screener .
   
   # Run the container
   docker run -u "$(id -u):$(id -g)" -e HOME=/app -p 8000:8000 --env-file .env mcp-thematic-screener
   ```

4. **Configure the MCP**:
In cursor: Go to File > Preferences > Cursor Preferences > MCP -> New MCP Server and add the following configuration:
```
"mcp-thematic-screener: {
    "url": "http://10.2.1.130:8000/mcp/"
}
```
![](./assets/cursor-mcp.png)

5. **Ask the agent to create a watchlist or generate a screening report for you**
For example, you can use the following prompt:
```
Create a watchlist called Next Generation Defense with the following companies: 3M Co., Accenture PLC, Alphabet Inc., BAE Systems PLC, Cisco Systems Inc., Elbit System Ltd., Gen Digital Inc., General Dynamics Corp., GM (General Motors Co.), and IBM Corp. Then, screen the companies in this watchlist for the theme Next Generation Defense for fiscal year 2024
```

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
   cd "Research_Tools_MCP_Integration"
   ```


3. **Set up credentials**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your credentials:
     ```
     BIGDATA_API_KEY=your_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

3. **Run the MCP server**:
   ```bash
   uv run MCP_Integration.py
   ```

4. **Configure the MCP**:
In cursor: Go to File > Preferences > Cursor Preferences > MCP -> New MCP Server and add the following configuration:
```
"mcp-thematic-screener: {
    "url": "http://10.2.1.130:8000/mcp/"
}
```
![](./assets/cursor-mcp.png)

5. **Ask the agent to create a watchlist or generate a screening report for you**
For example, you can use the following prompt:
```
Create a watchlist called Next Generation Defense with the following companies: 3M Co., Accenture PLC, Alphabet Inc., BAE Systems PLC, Cisco Systems Inc., Elbit System Ltd., Gen Digital Inc., General Dynamics Corp., GM (General Motors Co.), and IBM Corp. Then, screen the companies in this watchlist for the theme Next Generation Defense for fiscal year 2024
```
## Additional Examples of Usage

### Example prompt to create a watchlist and screen companies using the MCP integration

```txt
Create a watchlist called Next Generation Defense with the following companies: 3M Co., Accenture PLC, Alphabet Inc., BAE Systems PLC, Cisco Systems Inc., Elbit System Ltd., Gen Digital Inc., General Dynamics Corp., GM (General Motors Co.), and IBM Corp. Then, screen the companies in this watchlist for the theme Next Generation Defense for fiscal year 2024 and write the report as a markdown.
```

### Example curl commands to create a watchlist and screen companies using the MCP integration.
- Create a watchlist
```bash
curl -X POST http://localhost:8000/mcp   -H "Content-Type: application/json"   -H "Accept: application/json, text/event-stream"   -d '{"method": "tools/call", "params": {"name": "create_watchlist", "arguments": {"companies": ["3M Co.", "Accenture PLC", "Alphabet Inc.", "BAE Systems PLC", "Cisco Systems Inc.", "Elbit System Ltd.", "Gen Digital Inc.", "General Dynamics Corp.", "GM (General Motors Co.)", "IBM Corp."], "watchlist_name": "Next Generation Defense"}}, "jsonrpc": "2.0", "id": 2}'
```
- Screen companies in a watchlist
```bash
curl -X POST http://localhost:8000/mcp   -H "Content-Type: application/json"   -H "Accept: application/json, text/event-stream"   -d '{"method": "tools/call", "params": {"name": "screen_companies", "arguments": {"watchlist_name": "Next Generation Defense", "main_theme": "Next Generation Defense", "fiscal_year": 2024}}, "jsonrpc": "2.0", "id": 2}'
```

