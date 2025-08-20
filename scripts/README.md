# Scripts

This folder contains one-off scripts to run each stage of the pipeline.

- `01_collect_hindi.py` — Download and collate Hindi datasets.
- `02_clean_dedupe.py` — Clean and deduplicate raw datasets.
- `03_tokenize_shard.py` — Tokenize datasets and split into shards.
- `04_stats_curves.py` — Compute stats and prepare scaling law plots.

Scripts are designed to be executed step by step.  
Reusable utility functions live in `../src/`.
