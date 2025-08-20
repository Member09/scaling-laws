import os
import json
from pathlib import Path
from typing import Optional, Iterable, Dict, Any
from tqdm import tqdm
import argparse

from datasets import load_dataset

RAW_DIR = Path("../data/raw")

def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], total: Optional[int] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, total=total, desc=f"Writing {path.relative_to(RAW_DIR)}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def take(iterable, limit: Optional[int]):
    if limit is None:
        for x in iterable:
            yield x
    else:
        n = 0
        for x in iterable:
            yield x
            n += 1
            if n >= limit:
                break

# -------------------------
# Samanantar (parallel en↔hi). In HF config "hi", fields are typically: idx, src, tgt
# We'll assume src≈English, tgt≈Hindi for the "hi" config (common in this dataset variant).
# -------------------------
def collect_samanantar(limit: Optional[int], streaming: bool):
    ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=streaming)
    # Parallel file
    out_parallel = RAW_DIR / "samanantar" / "en_hi_parallel.jsonl"
    def gen_parallel():
        for ex in ds:
            src = (ex.get("src") or "").strip()
            tgt = (ex.get("tgt") or "").strip()
            if src and tgt:
                yield {"src_lang":"en","tgt_lang":"hi","src":src,"tgt":tgt}
    write_jsonl(out_parallel, take(gen_parallel(), limit=limit), total=None if streaming else None)

    # Monolingual Hindi (tgt side only)
    # Reload iterator if streaming, otherwise we can reuse ds
    ds_hi = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=streaming)
    out_hi = RAW_DIR / "samanantar" / "hi_mono.jsonl"
    def gen_hi():
        for ex in ds_hi:
            tgt = (ex.get("tgt") or "").strip()
            if tgt:
                yield {"lang":"hi","text":tgt}
    write_jsonl(out_hi, take(gen_hi(), limit=limit), total=None if streaming else None)

# -------------------------
# Wikipedia (Hindi monolingual)
# HF schema: id, url, title, text
# -------------------------
def collect_wikipedia(limit: Optional[int], streaming: bool, snapshot: str = "20231101.hi"):
    ds = load_dataset("wikimedia/wikipedia", snapshot, split="train", streaming=streaming)
    outpath = RAW_DIR / "wikipedia" / "hi_mono.jsonl"
    def gen():
        for ex in ds:
            text = (ex.get("text") or "").strip()
            if text:
                yield {
                    "lang":"hi",
                    "text": text,
                    "title": (ex.get("title") or ""),
                    "url": (ex.get("url") or "")
                }
    write_jsonl(outpath, take(gen(), limit=limit), total=None if streaming else None)

# -------------------------
# WikiLingua (Hindi summarization)
# Common HF configs are "wiki_lingua" or "wikilingua"; language keys vary.
# Your snippet shows split train with features ['url','article'] – some mirrors store summary embedded in article dict.
# We handle both common schemas below.
# -------------------------
def _to_str(x):
    """
    Normalize WikiLingua fields to a single string.
    Handles:
      - str
      - list/tuple of strings (Sequence)
      - dict with 'text' -> str or list
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        parts = [t.strip() for t in x if isinstance(t, str) and t.strip()]
        return " ".join(parts) if parts else None
    if isinstance(x, dict):
        v = x.get("text")
        return _to_str(v)
    return None


def collect_wikilingua(limit: Optional[int], streaming: bool):
    """
    Collect WikiLingua Hindi summarization pairs.
    Writes: data/raw/wikilingua/hi_sum.jsonl with records:
      {"lang":"hi","article":"...","summary":"...","url":"..."}
    """
    # Try the common builder names/configs in order
    ds = None
    tried = []
    for builder, config in [
        ("wiki_lingua", "hindi"),    # most common
        ("wikilingua", "hi"),        # some mirrors
        ("wiki_lingua", "hi"),       # alt spelling
    ]:
        try:
            ds = load_dataset(builder, config, split="train", streaming=streaming)
            break
        except Exception as e:
            tried.append(f"{builder}/{config}: {e}")
            ds = None

    if ds is None:
        raise RuntimeError(
            "Could not load WikiLingua Hindi. Tried:\n  - " + "\n  - ".join(tried)
        )

    outpath = RAW_DIR / "wikilingua" / "hi_sum.jsonl"

    def gen():
        for ex in ds:
            url = ex.get("url", "")

            # Common field names observed in WikiLingua variants:
            #   article: Sequence[str] OR {'text': Sequence[str]} OR str
            #   summary: Sequence[str] OR {'text': Sequence[str]} OR str
            article = _to_str(ex.get("article"))
            summary = _to_str(ex.get("summary"))

            # Some mirrors use 'highlights'/'summary_text' instead of 'summary'
            if not summary:
                summary = _to_str(ex.get("highlights")) or _to_str(ex.get("summary_text"))

            # Some very minimal mirrors only have 'article'; skip those entries
            if article and summary:
                yield {"lang": "hi", "article": article, "summary": summary, "url": url}

    write_jsonl(outpath, take(gen(), limit=limit), total=None if streaming else None)

# -------------------------
# Indic LLM corpus (monolingual, mixed Hindi + related languages)
# Raw dump only; filtering to "true Hindi" should be done in your cleaning stage.
# -------------------------
def collect_indicllm(limit: Optional[int], streaming: bool):
    ds = load_dataset("Hindi-data-hub/odaigen_hindi_pre_trained_sp", split="train", streaming=streaming)
    outpath = RAW_DIR / "indicllm" / "hi_mixed.jsonl"
    def gen():
        for ex in ds:
            # Common fields: 'text' / 'content'
            txt = ex.get("text") or ex.get("content") or ""
            txt = txt.strip()
            if txt:
                yield {"lang":"hi_like","text":txt,"note":"unfiltered_mixed_hindi_family"}
    write_jsonl(outpath, take(gen(), limit=limit), total=None if streaming else None)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_stream", action="store_true", help="Disable streaming mode (default: streaming enabled).")
    ap.add_argument("--limit", type=int, default=None, help="Take only N records per file (for smoke tests).")
    ap.add_argument("--sources", nargs="+", default=["samanantar","wikipedia","wikilingua"],
                    choices=["samanantar","wikipedia","wikilingua","indicllm"],
                    help="Which sources to collect.")
    args = ap.parse_args()

    streaming = not args.no_stream
    limit = args.limit

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if "samanantar" in args.sources:
        print("→ Collecting Samanantar (en↔hi parallel + hi mono)…")
        collect_samanantar(limit=limit, streaming=streaming)

    if "wikipedia" in args.sources:
        print("→ Collecting Wikipedia (hi mono)…")
        collect_wikipedia(limit=limit, streaming=streaming)

    if "wikilingua" in args.sources:
        print("→ Collecting WikiLingua (hi summarization)…")
        collect_wikilingua(limit=limit, streaming=streaming)

    if "indicllm" in args.sources:
        print("→ Collecting Indic LLM (raw mixed hi‑family)…")
        collect_indicllm(limit=limit, streaming=streaming)

    print("✅ Done. Raw files under data/raw/")

if __name__ == "__main__":
    main()