# Corpus snapshot

Built by `python -m ingest.crawl` from the pages in `ingest/sources.yaml`.

- `chunks.jsonl` — the chunks the index is built from (text + source URL, page title, section)
- `pages.jsonl` — every page crawled, with the time it was fetched

Commit these two files after each crawl so evals are reproducible, and note the crawl
date in your results. Campus policy pages change (ISSS updated CPT rules in July 2025),
so re-crawl before each semester and re-run the evals. `raw/` (cached HTML) is gitignored.
