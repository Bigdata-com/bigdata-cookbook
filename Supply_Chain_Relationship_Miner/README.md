# The suppliers of a company (notebook)

[supply_chain_relationships.ipynb](supply_chain_relationships.ipynb) finds the suppliers of a
company (the companies that sell it components, manufacturing, capacity, software or services) in
SEC filings, earnings calls, press releases and news, with Bigdata.com search and an OpenAI model,
following RavenPack's current supply-chain knowledge-graph approach. The example is NVIDIA over the
last year.

Every supplier comes with its evidence: the sentence that states the supply, copied from the text,
and the document, date, link and chunk it comes from. It also gives the number of records,
documents and sources behind it, and an explanation. The notebook is self-contained: the search,
the prompt, the checks and the selection of the suppliers are all in it, with the configuration
files in `config/`.

## How it works

1. **Search.** 61 search sentences, each written like a text that states a business relationship,
   are searched in Bigdata.com for the company, one search per month, over SEC filings, earnings
   calls and other transcripts, press releases and news. They look for every kind of relationship,
   not only supplies: a supply is often told as a partnership or a deal.
2. **Choose the chunks.** Every filing, transcript and press release chunk found is read, plus a
   fixed 10% of the news chunks (news is about 90% of what the search finds for NVIDIA). A chunk is
   read only when it names a pair of companies and one of them is the chosen company.
3. **Classify.** An OpenAI model reads one chunk at a time and answers, in a strict JSON format,
   which relationships between the companies it names the text states: a supply, a partnership, a
   sale or competition. Each answer has the type, the direction, an evidence code (components,
   contract manufacturing, software and cloud...) and the sentence that states it.
4. **Resolve names.** Companies the model finds in the text that Bigdata.com did not tag are looked
   up by name in the Bigdata.com Knowledge Graph.
5. **Keep the suppliers.** Every answer is checked (the companies are those of the chunk, the
   evidence code fits the type, the quote is there...). Each supply to the chosen company becomes a
   record, and the records of each supplier give it a status, its counts and an explanation. A
   supplier seen only in news is kept only when its publishers are trusted enough, by their
   Bigdata.com rank (the news rule, below). The partnerships, sales and competition the model also
   finds are left out.

The notebook follows these steps in sections 3 to 7; section 8 shows the suppliers and section 9
lists the limitations.

## What you get

In `output/` (`<LABEL>` is the short name set in section 1, `NVDA` in the example):

| File | One row per | Contents |
|---|---|---|
| `<LABEL>_suppliers.csv` | supplier of the chosen company | status; records, documents and sources; the confidence of its news publishers; explanation and link |
| `<LABEL>_supplier_records.csv` | piece of evidence (the grounding) | the quoted sentence, evidence code, document, publisher and its rank, date, headline, URL and chunk id |
| `<LABEL>_held_out.csv` | supply seen only in news that the news rule leaves out | the two companies, the confidence and ranks of its publishers, records, explanation and link |
| `<LABEL>_suppliers.png` | | a chart with the chosen company in the centre and every supplier around it |

Some terms:

- **Supply.** Another company `SUPPLIER_TO` the chosen company (it sells to it), or the chosen
  company `CUSTOMER_OF` another company (it buys from it), as a filing or a call puts it.
- **Record.** One piece of evidence of a supply: one chunk of one document. `records`, `documents`
  and `sources` (different publishers) count them.
- **1st class.** A record from an SEC filing, an earnings call or another transcript. Press releases
  and news are the 2nd class.
- **Status.** `ACTIVE` from the first record; `INACTIVE` when a record states the end, or after 12
  months without a new record.
- **News rule.** A record from a filing, a call, a transcript or a press release is enough for a
  supplier to be kept. One seen only in news needs trusted publishers: each publisher counts once
  and weighs by its Bigdata.com rank (`RANK_1`, the highest quality, and `RANK_2`: 1; `RANK_3`: 0.5;
  `RANK_4` and `RANK_5`: 0.2). The sum, at most 1, must reach 1: one rank 1-2 publisher, two rank 3
  or five rank 4-5.
- **News confidence.** `news_confidence` is that sum for a supplier's news publishers (empty with no
  news). The `confidence` of a record in `<LABEL>_supplier_records.csv` is the model's.
- **Explanation.** A sentence written from what is stored (no model writes it), with the quote of
  the supplier's best record: from a filing or a call when there is one, else the latest.
- **Named or described.** A supplier the text names, or one it only describes ("a leading Asian
  memory maker"). A described supplier gets an id of its own (`UNK_...`) and is left out of the
  chart.

More than half of the suppliers have a single record. Filter on `records >= 2` or `first_class`
for a smaller, more certain list.

## The example: NVIDIA, 2025-11-16 to 2026-11-15

The outputs saved in the notebook come from this run:

| Step | Result |
|---|---|
| Search | 826 searches; 49,226 documents and 60,608 chunks |
| Chunks read by the model | 8,255 (all the filings, transcripts and press releases; 10% of the news) |
| Supplies to NVIDIA | 834 records; 254 relationships kept, 19 left out by the news rule |
| News publishers of the supplies | 142: 20 rank 1, 30 rank 2, 77 rank 3, 15 rank 4 |
| Suppliers of NVIDIA | 158 by name (25 more only described); 63 with a record from a filing or a call, 64 with two records or more |

The suppliers with most records are TSMC, SK Hynix, Samsung Electronics, Samsung Electro-Mechanics
and Amkor.

The news rule leaves out 19 supplies seen only in news: 16 with a single rank 3 publisher
(`news_confidence` 0.5) and 3 with a single rank 4-5 publisher (0.2). Without the rule
(`NEWS_RULE = "sources"`, `NEWS_MIN_SOURCES = 1`) NVIDIA has 176 suppliers by name.

In the saved run, all 8,255 model answers and the ranks of the 142 news publishers came from the
cache of earlier runs (no OpenAI call). A run from scratch makes all the calls (see below).

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
| `NEWS_RULE` | `"rank"` | which suppliers seen only in news are kept: `"rank"`, by the rank of their publishers, or `"sources"`, by their number |
| `NEWS_MIN_CONFIDENCE` | 1.0 | the rank rule's threshold (more than 0, at most 1) |
| `NEWS_MIN_SOURCES` | 1 | the sources rule's threshold: the different publishers a supply seen only in press releases and news needs (1: every supply is kept) |
| `INSTITUTIONS`, `CENTRAL_BANKS` | `True`, `False` | whether governments, agencies, universities... and central banks can be counterparties (a university or a national lab can be a supplier) |
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

The search returns each publisher's rank. A news publisher without one is looked up with one
Bigdata.com search, once (a search cached before the rank was kept has none: for NVIDIA, 142
publishers and about a minute).

Everything is cached in `cache/`: the search (per company and dates), the company names, the
publisher ranks looked up, and the model answers (per model and prompt). A run that stops resumes
where it stopped, and a rerun with the same parameters costs nothing. New dates search again, but
the model is only called for chunks it has not answered; a larger share also calls it only for the
new chunks. Changing `MODEL` asks the model again about every chunk. Delete `cache/` to start from
scratch.

## Limitations

- **Only what the text states.** Co-mentions, rankings, ownership and M&A, lending, litigation,
  speculation and market commentary are not recorded.
- **Model noise.** Asking the model again about the same chunks changes about 7% of the pairs,
  almost all of them with a single record.
- **News is sampled.** 10% of the news chunks are read. The choice is fixed per chunk, so a larger
  share reads the same chunks plus new ones.
- **The news rule trusts the publisher, not the text.** The rank rates a publisher, not an article:
  a rank 1 publisher can carry a rumour and a rank 3 one an official announcement. In RavenPack's
  checks, most relationships the rule leaves out are stated in the text, and since only 10% of the
  news is read, a supplier with a single rank 3 publisher may have others in the news not read.
  They are all in `<LABEL>_held_out.csv`; `NEWS_RULE = "sources"` with `NEWS_MIN_SOURCES = 1` keeps
  them.
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
| `cache/` | written by the notebook: search results, names, publisher ranks and model answers (a rerun reuses them) |
| `output/` | written by the notebook: the results as CSV files and the chart |
