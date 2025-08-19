from datasets import load_dataset
from pathlib import Path
from typing import Optional, Iterable
from tqdm import tqdm
import json
import argparse
import os

'''
1. Samanantar
	•	Origin: IIT Bombay + AI4Bharat release (2021)
	•	Type: Largest publicly available parallel dataset for English ↔ Indic languages (including Hindi).
	•	Size: For Hindi, ~46M parallel sentence pairs (~1B tokens total combined).
	•	Sources: Web crawls, news, government documents, Wikipedia.
	•	License: CC BY 4.0.
    Source.                          Type / Description                                            Estimated Size
Samanantar (AI4Bharat)           English-Hindi parallel corpus                             Hindi side only; ~25-50M sentences 

2. Indic LLM

3. OPUS

4. Wikipedia

'''
RAW_DIR = Path("data/raw")

def write_jsonl(path: Path, records: Iterable[dict], total: Optional[int] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, total=total, desc=f"Writing {path.name}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_text(x) -> Optional[str]:
    """
    Normalize a record to plain text string.
    Returns None if no text is found.
    """
    # Common fields in HF text datasets
    if isinstance(x, dict):
        # Wikipedia HF schema

        for k in ("text", "content"):
            if k in x and isinstance(x[k], str):
                return x[k].strip()
        
        if "tgt" in x and isinstance(x["tgt"], str):
            return x["tgt"].strip()
    return None

def iter_text(ds_split, limit: Optional[int] = None):
    n = 0
    for ex in ds_split:
        txt = normalize_text(ex)
        if txt:
            yield {"text": txt}
            n += 1
            if limit and n >= limit:
                break



def collect_samanantar(limit: Optional[int], streaming: bool):
    """ Dataset source : https://huggingface.co/datasets/ai4bharat/samanantar
    """
    ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=streaming)
    # ds : Dataset({
    #     features: ['idx', 'src', 'tgt'],
    #     num_rows: 10125706
    # })
    outpath = RAW_DIR / "samanantar_hi.jsonl"
    write_jsonl(outpath, iter_text(ds, limit=limit), total=None if streaming else (len(ds) if not limit else (min(len(ds), limit))))

def collect_indicLLM(limit: Optional[int], streaming: bool):
    """Dataset source : https://arxiv.org/abs/2407.09855 -> https://huggingface.co/datasets/Hindi-data-hub/odaigen_hindi_pre_trained_sp
       Huge dataset but it is a mix of all hindi languages, includes kannauji, marathi, sanskrit, himachali, awadhi, bhili etc...
       TODO : clean this to contain hindi text only, o/w don't include.
    """
    ds = load_dataset("Hindi-data-hub/odaigen_hindi_pre_trained_sp",split="train", streaming=streaming)
    outpath = RAW_DIR / "indicllm_hi.jsonl"
    write_jsonl(outpath, iter_text(ds, limit=limit), total= None if streaming else (len(ds) if not limit else min(limit, len(ds))))


def collect_OSCAR():
    # ds = load_dataset("oscar-corpus/OSCAR-2301", "hi", split="train", use_auth_token=True, streaming=True)
    # write_to_file(hindi_texts=hindi_texts)
    return


def collect_wiki(limit: Optional[int], streaming: bool):
    """ Dataset source : https://huggingface.co/datasets/wikimedia/wikipedia
    """
    ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=streaming)
    # ds_wiki : IterableDataset({
    #     features: ['id', 'url', 'title', 'text'],
    #     num_shards: 2
    # })
    outpath = RAW_DIR / "wikipedia_hi.jsonl"
    write_jsonl(outpath, iter_text(ds_wiki, limit=limit), total= None if streaming else (len(ds_wiki) if not limit else min(limit, len(ds_wiki))))

# def write_to_file(hindi_texts: list, path: Path = RAW_DIR):
#     # Save to file
#     with open("data/raw/hindi_samanantar.txt", "w") as f:
#         for line in hindi_texts:
#             f.write(line.strip() + "\n")

def main():

    print("→ Collecting Samanantar (Hindi side)…")
    # Collect data from Samanantar
    collect_samanantar(10, True)

    # # Collect data from mc4
    # collect_mC4()

    # # Collect data from OSCAR
    # collect_OSCAR()
    print(f"→ Collecting Wikipedia…")
    # Collect data from wiki
    collect_wiki(10, True)

    # print(f"→ Collecting Indic LLM")
    # collect_indicLLM(10, True)
    print("✅ Done. Files are in data/raw/*.jsonl")


if __name__ == "__main__":
    main()