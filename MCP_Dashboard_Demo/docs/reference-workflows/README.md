# Reference GitHub Actions (not active here)

These YAML files are **documentation only**. They are **not** loaded by GitHub from this path.

This cookbook folder is a **source illustration** for the MCP → `GROUNDED_DATA` → React pattern. **Deployment and scheduled refresh** should live in a **separate repository** with its own `.github/workflows/`.

## Update cycle and agents

The illustration repo favors **Cursor** as the interactive trigger (agent + Bigdata.com MCP → edit `src/dashboard.jsx`). **Claude Code** or another MCP-capable agent is an equivalent choice. [`update.yml`](update.yml) is a stub: replace the placeholder step with your API or headless agent invocation if you want GitHub-hosted runs instead of editor-driven cycles.

## How to use

1. Copy the desired workflow into your deployment repo as `.github/workflows/<name>.yml`.
2. **Uncomment** the entire `on:` block at the top of the file (triggers are commented out here so nothing runs if the file is ever misplaced).
3. Add the required secrets (e.g. `FLY_API_TOKEN`, `ANTHROPIC_API_KEY`, `BIGDATA_API_KEY`) in the deployment repo’s GitHub Settings → Secrets.

## Files

| File | Purpose |
|------|---------|
| [`deploy.yml`](deploy.yml) | Build and deploy to Fly.io on push to `main` (when uncommented). |
| [`update.yml`](update.yml) | Placeholder “update cycle” job; extend with your LLM/MCP automation (when uncommented). |
