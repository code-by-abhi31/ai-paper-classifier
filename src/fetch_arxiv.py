# src/fetch_arxiv.py
# Simple arXiv metadata + abstract fetcher that writes JSONL to data/
import arxiv
import json
import time
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def safe_author_name(a):
    # arxiv.Author has .name in most versions; fallback to str()
    return getattr(a, "name", str(a))

def fetch_arxiv(query="machine learning", max_results=100, out_fname=None, sleep_per_item=0.2):
    """Fetch metadata from arXiv and save as JSONL.
    - query: arXiv search query string
    - max_results: number of papers to fetch
    - out_fname: Path or str for output file (optional)
    - sleep_per_item: polite delay between handling items
    """
    if out_fname is None:
        safe_q = query.replace(" ", "_")[:50]
        out_fname = DATA_DIR / f"arxiv_{safe_q}_{max_results}.jsonl"
    else:
        out_fname = Path(out_fname)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    with out_fname.open("w", encoding="utf8") as fh:
        for i, result in enumerate(search.results(), start=1):
            item = {
                "id": result.entry_id,
                "title": (result.title or "").strip(),
                "authors": [safe_author_name(a) for a in result.authors],
                "summary": (result.summary or "").strip(),   # abstract
                "published": result.published.isoformat() if getattr(result, "published", None) else None,
                "updated": result.updated.isoformat() if getattr(result, "updated", None) else None,
                "primary_category": getattr(result, "primary_category", None),
                "categories": getattr(result, "categories", []),
                "pdf_url": getattr(result, "pdf_url", None),
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

            # polite pause so we don't hammer anything
            time.sleep(sleep_per_item)

            # small progress print
            if i % 10 == 0:
                print(f"  fetched {i} / {max_results} items...")

    print(f"Wrote {out_fname} ({max_results} items requested)")
    return out_fname

if __name__ == "__main__":
    # Default run: 100 machine-learning papers. Change query/max_results as you like.
    fetch_arxiv(query="machine learning", max_results=100)
