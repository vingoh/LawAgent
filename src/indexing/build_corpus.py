"""
Build unified corpus from court_considerations.csv and laws_de.csv.

Outputs:
  indexes/corpus.pkl          - List[dict] with keys: citation, text, source, indexed_text
  indexes/citation_to_idx.pkl - dict: citation str -> corpus list index
"""

import csv
import os
import pickle
import re
from tqdm import tqdm

DATASET_DIR = os.path.join(os.path.dirname(__file__), "../../dataset")
INDEX_DIR   = os.path.join(os.path.dirname(__file__), "../../indexes")

COURT_CSV = os.path.join(DATASET_DIR, "court_considerations.csv")
LAWS_CSV  = os.path.join(DATASET_DIR, "laws_de.csv")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _has_numeric_sr_code(citation: str) -> bool:
    """Return True for laws_de rows that use numeric SR codes, e.g. 'Art. 1 112'.
    These cannot match gold citations which use abbreviations (OR, ZGB, ...).
    """
    return bool(re.search(r'\s[\d.]+$', citation))


def _is_malformed_consideration(citation: str) -> bool:
    """Return True for court consideration citations with date-embedded noise.
    e.g. 'BGE 139 I 2 E. 1.12.2011' — any segment of 4 digits after E. is a year.
    Logs a warning but does NOT filter outright; caller decides based on train check.
    """
    m = re.search(r'E\.\s*([\d.]+)$', citation)
    if m:
        parts = m.group(1).split('.')
        if any(len(p) == 4 for p in parts):
            return True
    return False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_laws(path: str) -> list[dict]:
    """Load laws_de.csv, filter numeric SR codes, build indexed_text."""
    rows = []
    skipped = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading laws_de"):
            citation = row["citation"].strip()
            if _has_numeric_sr_code(citation):
                skipped += 1
                continue
            title = row.get("title", "").strip()
            text  = row["text"].strip()
            indexed_text = " ".join(filter(None, [citation, title, text]))
            rows.append({
                "citation":     citation,
                "text":         text,
                "source":       "law",
                "indexed_text": indexed_text,
            })
    print(f"laws_de: loaded {len(rows):,}, skipped {skipped:,} (numeric SR codes)")
    return rows


def load_court(path: str) -> list[dict]:
    """Load court_considerations.csv, filter malformed citations,
    deduplicate by keeping the longest text per citation key.
    """
    grouped: dict[str, dict] = {}
    malformed = 0

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading court_considerations"):
            citation = row["citation"].strip()

            if _is_malformed_consideration(citation):
                malformed += 1
                continue

            text = row["text"].strip()
            if citation not in grouped or len(text) > len(grouped[citation]["text"]):
                grouped[citation] = {"citation": citation, "text": text}

    rows = []
    for cit, d in grouped.items():
        indexed_text = cit + " " + d["text"]
        rows.append({
            "citation":     cit,
            "text":         d["text"],
            "source":       "court",
            "indexed_text": indexed_text,
        })

    print(f"court_considerations: loaded {len(rows):,} unique citations, "
          f"skipped {malformed:,} malformed")
    return rows


# ---------------------------------------------------------------------------
# Coverage check against val gold citations
# ---------------------------------------------------------------------------

def check_val_coverage(corpus_index: dict[str, int]) -> None:
    val_path = os.path.join(DATASET_DIR, "val.csv")
    if not os.path.exists(val_path):
        return
    all_gold, missing = [], []
    with open(val_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for c in row["gold_citations"].split(";"):
                c = c.strip()
                if c:
                    all_gold.append(c)
                    if c not in corpus_index:
                        missing.append(c)
    total = len(all_gold)
    covered = total - len(missing)
    print(f"\nVal gold coverage: {covered}/{total} ({100*covered/total:.1f}%)")
    if missing:
        print(f"Missing ({len(missing)}):")
        for m in missing:
            print(f"  {m}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)

    laws_docs  = load_laws(LAWS_CSV)
    court_docs = load_court(COURT_CSV)

    corpus: list[dict] = laws_docs + court_docs
    print(f"\nTotal corpus size: {len(corpus):,} documents")

    citation_to_idx: dict[str, int] = {
        doc["citation"]: i for i, doc in enumerate(corpus)
    }

    # Validate coverage against val gold
    check_val_coverage(citation_to_idx)

    # Save
    corpus_path = os.path.join(INDEX_DIR, "corpus.pkl")
    idx_path    = os.path.join(INDEX_DIR, "citation_to_idx.pkl")

    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(idx_path, "wb") as f:
        pickle.dump(citation_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved corpus      → {corpus_path}")
    print(f"Saved citation idx → {idx_path}")


if __name__ == "__main__":
    main()
