# Fetch Document API (Tutorial)

Fetch full documents from the Bigdata.com API. Use the **Fetch Document** endpoint (`GET /v1/documents/{document_id}`) when you need the complete processed content of a document — metadata, structured body, entities, sentences, analytics, and profiling — instead of search snippets.

**API Reference:** [docs.bigdata.com/api-reference/search/fetch-document](https://docs.bigdata.com/api-reference/search/fetch-document)

## What this tutorial covers

- Calling `GET /v1/documents/{document_id}` to fetch a document
- Understanding the `web_content` flag in the response
- Saving annotated document JSON to disk
- Extracting title, body text, analytics, events, and entity data from the response

## Response behavior

The endpoint always returns `{ url, web_content }`. The `url` is always a pre-signed URL (valid ~24 hours) that returns the full annotated document JSON when accessed.

| `web_content` | Meaning | Difference in downloaded JSON |
|---|---|---|
| `true` | Publicly accessible content (e.g. news articles) | `document.metadata.url` contains a link to the original web page |
| `false` | Premium / non-web content | `document.metadata` has no `url` field |

The notebook follows the pre-signed URL for every document. The downloaded JSON contains:
- `document` – metadata (source, timestamps, file info)
- `content` – title and body blocks (TEXT, TABLE, LIST_ORDERED, LIST_UNORDERED, HEADING, FOOTER)
- `profiling` – processor timestamps
- `analytics` – document-level metrics, events array, entities array

## Setup

**Prerequisites:** Python 3.8+, `requests`, and a Bigdata.com API key.

```bash
cd API_Tutorials/Document_Download_API
uv venv
uv add requests
```

If you don't use `uv`:

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

- Authentication is via the `X-API-KEY` header using `BIGDATA_API_KEY`.
- The notebook uses sample document IDs; replace them with your own document IDs from search or other APIs.
- Pre-signed URLs expire after ~24 hours; request a new one if needed.
