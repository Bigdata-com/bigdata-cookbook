# Supply-chain relationships of a company (notebook)

[supply_chain_relationships.ipynb](supply_chain_relationships.ipynb) finds the suppliers,
customers, partners and competitors of a company in SEC filings, earnings calls, press releases and
news, with Bigdata.com search and an OpenAI model, following RavenPack's current supply-chain
knowledge-graph approach. The example is NVIDIA over the last year.

Every relationship comes with its evidence: the sentence that states it, copied from the text, and
the document, date, link and chunk it comes from. It also gives the number of records, documents and
sources behind the relationship, and an explanation. The notebook is self-contained: the search, the
prompt, the checks and the graph are all in it, with the configuration files in `config/`.

## How it works

1. **Search.** 61 search sentences, each written like a text that states a business relationship,
   are searched in Bigdata.com for the company, one search per month, over SEC filings, earnings
   calls and other transcripts, press releases and news.
2. **Choose the chunks.** Every filing, transcript and press release chunk found is read, plus a
   fixed 10% of the news chunks (news is about 80% of what the search finds). A chunk is read only
   when it names a pair of companies and one of them is the chosen company.
3. **Classify.** An OpenAI model reads one chunk at a time and answers, in a strict JSON format,
   which relationships between the companies it names the text states. Each answer has the type,
   the direction, an evidence code (components, contract manufacturing, joint development...) and
   the sentence that states it.
4. **Resolve names.** Companies the model finds in the text that Bigdata.com did not tag are looked
   up by name in the Bigdata.com Knowledge Graph.
5. **Build the graph.** Every answer is checked (the companies are those of the chunk, the evidence
   code fits the type, the quote is there...). Each accepted relation becomes a record, and the
   records of a pair of companies and a type form a relationship with a status, its counts and an
   explanation.

The notebook follows these steps in sections 3 to 7; section 8 shows the results for the chosen
company and section 9 lists the limitations.

## What you get

In `output/`, as CSV files (`<LABEL>` is the short name set in section 1, `NVDA` in the example):

| File | One row per | Contents |
|---|---|---|
| `<LABEL>_counterparties.csv` | company related to the chosen company, and role | supplier, customer, partner or competitor; status; records, documents and sources; explanation and link |
| `<LABEL>_relationships.csv` | relationship of the graph | the two companies, type, status, dates, records, documents, sources, explanation with its quote and link |
| `<LABEL>_records.csv` | piece of evidence (the grounding) | the quoted sentence, evidence code, document, date, headline, URL and chunk id |
| `<LABEL>_rejections.csv` | relation the model or the checks rejected | the reason and the quoted text |
| `<LABEL>_related_companies.png` | | a chart with the chosen company in the centre and every related company around it |

Some terms:

- **Relationship types.** `SUPPLIER_TO` (the first company sells to the second), `CUSTOMER_OF` (the
  first buys from the second), `PARTNER_WITH` and `COMPETITOR_OF`.
- **Role.** What the other company is to the chosen company: a *supplier* sells to it, a *customer*
  buys from it. A company with two roles (a supplier that is also a partner) appears once per role.
- **Record.** One piece of evidence for a relationship: one chunk of one document. `records`,
  `documents` and `sources` (different publishers) count them.
- **1st class.** A record from an SEC filing, an earnings call or another transcript. Press releases
  and news are the 2nd class.
- **Status.** `ACTIVE` from the first record; `INACTIVE` when a record states the end, or after 12
  months without a new record.
- **Explanation.** A sentence written from what is stored (no model writes it), with the quote of
  the relationship's best record: from a filing or a call when there is one, else the latest.
- **Named or described.** A company the text names, or one it only describes ("a leading Asian
  memory maker"). A described company gets an id of its own (`UNK_...`) and is left out of the
  chart.

About two pairs of companies in three have a single record. Filter on `records >= 2` or
`first_class` for a smaller, more certain list.

## The example: NVIDIA, 2025-11-16 to 2026-11-15

The outputs saved in the notebook come from this run:

| Step | Result |
|---|---|
| Search | 826 searches; 49,226 documents and 60,608 chunks |
| Chunks read by the model | 8,255 (all the filings, transcripts and press releases; 10% of the news) |
| Graph | 5,489 relationships from 12,764 records; 6,516 relations rejected |
| Companies related to NVIDIA, by name | 946: 176 suppliers, 369 customers, 675 partners, 52 competitors |

The suppliers with most records are TSMC, SK Hynix, Samsung Electronics, Samsung Electro-Mechanics
and Amkor. The customers with most records are Microsoft, ASUSTeK, Lenovo, CoreWeave and SpaceX.

In the saved run, 8,208 of the 8,255 model answers came from the cache of an earlier run with the
same prompt and model, and 47 were new calls. A run from scratch makes all the calls (see below).

## Run it

Needs [uv](https://docs.astral.sh/uv/) 0.9.17 or later (`uv self update`; uv installs Python 3.12
itself if it is missing), a Bigdata.com API key and an OpenAI API key with access to the model set
in section 1 (`gpt-5.6-luna`).

```bash
uv sync --locked      # the exact versions of uv.lock
uv run jupyter lab    # then open supply_chain_relationships.ipynb and run all the cells
```

Set `BIGDATA_API_KEY` and `OPENAI_API_KEY` in the environment or in a `.env` file in this folder
(the notebook asks for them otherwise). The notebook also opens in VS Code with the `.venv` kernel.

## Change the company or the parameters

Section 1 of the notebook holds the parameters:

| Parameter | Default | What it does |
|---|---|---|
| `COMPANY_ID`, `LABEL` | `E09E2B`, `NVDA` | the company's Bigdata.com id and a short name for the files. To find an id, run `pd.DataFrame(kg.companies("Apple"))` in section 3 |
| `START`, `END` | the last year | the dates searched, both days included |
| `SHARE_1ST_CLASS`, `SHARE_PRESS_RELEASES`, `SHARE_NEWS` | 1.0, 1.0, 0.1 | the share of the chunks found that the model reads, by source type |
| `NEWS_LANGUAGES` | all | e.g. `["en"]`: only news first published in English |
| `NEWS_MIN_SOURCES` | 1 | 2: a relationship found only in news needs two different publishers |
| `INSTITUTIONS`, `CENTRAL_BANKS` | `True`, `False` | whether governments, agencies, universities... and central banks can be counterparties |
| `MODEL`, `LLM_WORKERS` | `gpt-5.6-luna`, 8 | the OpenAI model and the calls in parallel |
| `MAX_LLM_CALLS` | `None` | a cap on the OpenAI calls, e.g. 100 for a quick, cheap test |

## Time and cost

For the NVIDIA example from scratch, approximately:

- **Search:** about 15 minutes and about 830 Bigdata.com searches.
- **Classification:** about 8,300 OpenAI calls, about 2 h 45 min and about 8 USD (measured on
  2026-11-23 with `gpt-5.6-luna` and 8 workers: about 50 calls per minute, about 1 USD per 1,000
  calls). OpenAI's rate limits and prices decide the real figures.

A company with less coverage needs fewer calls. Section 5 prints the estimate for your parameters
before it calls OpenAI.

Everything is cached in `cache/`: the search (per company and dates), the company names, and the
model answers (per model and prompt). A run that stops resumes where it stopped, and a rerun with
the same parameters costs nothing. New dates search again, but the model is only called for chunks
it has not answered; a larger share also calls it only for the new chunks. Changing `MODEL` asks the
model again about every chunk. Delete `cache/` to start from scratch.

## Limitations

- **Only what the text states.** Co-mentions, rankings, ownership and M&A, lending, litigation,
  speculation and market commentary are not recorded.
- **Model noise.** Asking the model again about the same chunks changes about 7% of the pairs,
  almost all of them with a single record.
- **News is sampled.** 10% of the news chunks are read. The choice is fixed per chunk, so a larger
  share reads the same chunks plus new ones.
- **Names found by the model.** About 80% are found in the Knowledge Graph, and short names and
  acronyms are sometimes resolved to the wrong company. In the example, "SPIL" (Siliconware Precision
  Industries) became Sun Pharmaceutical Industries, which appears as a supplier of NVIDIA.
- **Search limits.** A search returns at most 1,000 chunks; the search cell reports how many
  searches were cut (6 of 826, all in news, for NVIDIA).

## Files

| Path | Contents |
|---|---|
| `supply_chain_relationships.ipynb` | the notebook |
| `config/` | the search sentences, relationship rules, evidence codes and counterparty lists of the approach |
| `pyproject.toml`, `uv.lock` | the Python environment |
| `cache/` | written by the notebook: search results, names and model answers (a rerun reuses them) |
| `output/` | written by the notebook: the results as CSV files and the chart |
