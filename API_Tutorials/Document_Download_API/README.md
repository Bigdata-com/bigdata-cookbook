# Document Download API (Tutorial)

Download full documents from the Bigdata.com API. Use the **Document Download API** when you need the complete processed content of a document (metadata, structured body, entities, sentences) instead of search snippets.

## What this tutorial covers

- Calling `GET /documents/<id>` to fetch a full document
- Handling both response types: **direct JSON** (documents &lt; 5MB) and **pre-signed URL** (documents ≥ 5MB)
- Saving documents to disk (JSON and plain text)
- Extracting title and body text from the response

## Response behavior

| Document size | API returns | What the notebook does |
|---------------|-------------|------------------------|
| &lt; 5MB      | JSON in the response body | Uses it directly |
| ≥ 5MB         | JSON with a `url` field (pre-signed S3 link) | Follows the URL and downloads the JSON |

The notebook handles both cases so you don’t have to branch in your own code.

## Setup

**Prerequisites:** Python 3.8+, `requests`, and a Bigdata.com API key.

```bash
cd Document_Download_API
uv venv
uv add requests
```

If you don’t use `uv`:

```bash
pip install requests
```

Set your API key (e.g. in `.env` or your shell):

```bash
export BIGDATA_API_KEY=your_api_key_here
```

You can create an API key at [platform.bigdata.com/api-keys](https://platform.bigdata.com/api-keys).

## Run

```bash
uv run jupyter notebook document_download.ipynb
```

Or:

```bash
jupyter notebook document_download.ipynb
```

## Output (typical)

- **`output/`** – Downloaded documents as JSON (and optional `.txt` with title + body only)

## Notes

- Authentication is via the `x-api-key` header using `BIGDATA_API_KEY`.
- The notebook uses sample document IDs; replace them with your own document IDs from search or other APIs.
