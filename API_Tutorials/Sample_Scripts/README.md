# Bigdata.com Sample Scripts

Sample scripts built on [Bigdata.com](https://bigdata.com) APIs.

Each folder is a single use-case with:
- one primary runnable script,
- local artifacts (examples, plots, outputs),
- a dedicated README.

## Quickstart

```bash
cd API_Tutorials/Sample_Scripts/<use_case_folder>
uv venv
uv pip install -r requirements.txt  # if present
export BIGDATA_API_KEY=your_api_key_here
uv run python <script_name>.py
```

## Use-case Index

| Use case | Folder | Primary script | APIs |
|---|---|---|---|
| Career network around a person | `career_network` | `people_network_analysis.py` | Co-mentions, Knowledge Graph |
| Apple co-mention map | `entity_comention_map` | `apple_comentions.py` | Co-mentions, Search, Knowledge Graph |
| Repsol spike and top chunks | `volume_spike_top_chunks` | `repsol_volume_spike.py` | Volume, Search, Knowledge Graph |
| Batch earnings sentiment | `earnings_sentiment_batch` | `earnings_sentiment.py` | Batch Search, Knowledge Graph |
| Nvidia vs AMD narrative comparison | `competitor_narrative_compare` | `nvidia_vs_amd.py` | Search, Knowledge Graph |
| Macro theme radar | `macro_theme_radar` | `macro_theme_radar.py` | Volume |
| Jensen Huang profiler | `person_in_the_news_profiler` | `jensen_huang_profiler.py` | Co-mentions, Search, Knowledge Graph |
| Daily briefing generator | `daily_briefing_generator` | `morning_briefing.py` | Batch Search, Knowledge Graph |
| Source tier benchmark | `source_quality_benchmark` | `fed_source_tier_comparison.py` | Search, Knowledge Graph |
| Geopolitical heatmap | `geopolitical_risk_heatmap` | `geopolitical_heatmap.py` | Volume |
| Quantum thematic universe | `thematic_universe_builder` | `quantum_computing_comentions.py` | Co-mentions, Knowledge Graph |
| PepsiCo co-mention network | `pepsico_comention_network` | `pepsi_comention_network.py` | Co-mentions, Knowledge Graph |
| Iran spike analysis | `iran_spike_analysis` | `iran_volume_spikes.py` | Volume, Co-mentions, Knowledge Graph |
| Company pulse (24h) | `company_pulse_24h` | `company_pulse.py` | Volume, Search, Knowledge Graph |
| Apple macro earnings extraction | `apple_macro_earnings` | `apple_macro_earnings.py` | Search, Knowledge Graph |


