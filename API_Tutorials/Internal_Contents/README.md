# Internal Contents — Upload & Search

Upload private PDFs to Bigdata.com, wait for enrichment and indexing, then search them using tag filters.

**API references:**
- [Enrich document](https://docs.bigdata.com/api-reference/documents/enrich-document)
- [Search tag filter](https://docs.bigdata.com/api-reference/search/search-documents#body-query-filters-tag)

## What this tutorial covers

- Request a pre-signed URL via `POST /contents/v1/documents`
- Upload PDF bytes with `PUT` to the pre-signed URL
- Poll `GET /contents/v1/documents/{id}` until `status` is `completed`
- Search uploaded files with `POST /v1/search` using `query.filters.tag` and the `my_files` category

## Sample files

| File | Display name | Tags |
|---|---|---|
| `sample_files/Boeing-Corporate-Actions.pdf` | Boeing Corporate Actions from 2024 to 2026 | Boeing, Corporate Actions, Workflow |
| `sample_files/Brazil_Economic_Analysis_2026.pdf` | Brazil Economic Analsis as of June 2026 | Brazil, Country, Macro Analysis, MCP |

## Setup

**Prerequisites:** Python 3.8+, `requests`, and a Bigdata.com API key.

```bash
cd API_Tutorials/Internal_Contents
pip install requests
export BIGDATA_API_KEY=your_api_key_here
```

Create an API key at [platform.bigdata.com/api-keys](https://platform.bigdata.com/api-keys).

## Run locally

```bash
jupyter notebook upload_and_search.ipynb
```

## Run on Google Colab

Open the notebook via the Colab badge at the top of `upload_and_search.ipynb`, then add `BIGDATA_API_KEY` to **Secrets** (key icon in the left sidebar).

## Notes

- PDF enrichment typically takes 1–3 minutes per file.
- Re-running the upload cells creates new documents each time; skip them on subsequent runs if you only want to test search.
- Authentication uses the `X-API-KEY` header with `BIGDATA_API_KEY`.
