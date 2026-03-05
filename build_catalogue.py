# =============================================================================
# HIVEMIND — Catalogue Builder
# Run this ONCE to build content_dict.pkl for app.py
# Fetches the full TMDB catalogue with rich tags (genres + cast + keywords +
# director + overview) using parallel requests for speed.
# =============================================================================

import requests
import pandas as pd
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY        = "8265bd1679663a7ea12ac168da84d2e8"
BASE           = "https://api.themoviedb.org/3"
MAX_WORKERS    = 8     # parallel threads for page fetching
DETAIL_WORKERS = 12    # parallel threads for per-item detail fetching
MIN_VOTES      = 200   # skip items with fewer votes (filters out obscure junk)
MIN_RATING     = 5.0   # skip items rated below this
MAX_PAGES      = 500   # TMDB hard cap per endpoint (each page = 20 items)
                       # 500 pages × 20 = 10,000 movies + 10,000 TV = ~20,000 total

HEADERS = {"User-Agent": "HivemindApp/1.0"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_get(url, params=None, retries=3):
    """GET with retries and timeout. Returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=8)
            if r.status_code == 429:          # rate limited — back off
                time.sleep(2 ** attempt)
                continue
            if r.ok:
                return r.json()
        except Exception:
            time.sleep(0.5)
    return None

# ── Step 1: Discover all pages ────────────────────────────────────────────────
def fetch_discover_page(content_type, page):
    """Fetch one page of discover results. Returns list of raw items."""
    data = safe_get(f"{BASE}/discover/{content_type}", params={
        "api_key":           API_KEY,
        "sort_by":           "popularity.desc",
        "vote_count.gte":    MIN_VOTES,
        "vote_average.gte":  MIN_RATING,
        "page":              page,
    })
    if not data:
        return [], 0
    title_key = "title" if content_type == "movie" else "name"
    items = []
    for item in data.get("results", []):
        items.append({
            "id":     item["id"],
            "title":  item.get(title_key, "Unknown"),
            "rating": round(item.get("vote_average", 0), 1),
            "type":   content_type,
            "tags":   item.get("overview", ""),   # placeholder — enriched below
        })
    return items, data.get("total_pages", 1)

def fetch_all_pages(content_type):
    """
    Fetch ALL available pages in parallel.
    First fetches page 1 to get total_pages, then fans out.
    """
    print(f"\n📡 Fetching {content_type} catalogue...")
    first_items, total_pages = fetch_discover_page(content_type, 1)
    total_pages = min(total_pages, MAX_PAGES)
    print(f"   → {total_pages} pages available for {content_type}")

    all_items = list(first_items)

    pages_to_fetch = range(2, total_pages + 1)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_discover_page, content_type, p): p for p in pages_to_fetch}
        for i, future in enumerate(as_completed(futures), 1):
            items, _ = future.result()
            all_items.extend(items)
            if i % 50 == 0:
                print(f"   → {i}/{len(pages_to_fetch)} pages done ({len(all_items)} items so far)")

    df = pd.DataFrame(all_items).drop_duplicates(subset=["id"])
    print(f"   ✓ {len(df)} unique {content_type}s fetched")
    return df

# ── Step 2: Enrich each item with rich tags ───────────────────────────────────
def build_rich_tags(item_id, content_type):
    """
    For a single item, fetch:
      - genres
      - top 5 cast members
      - director (movies) / creators (TV)
      - TMDB keywords
      - overview
    Returns a combined tag string.
    """
    # Single request gets details + credits + keywords
    data = safe_get(f"{BASE}/{content_type}/{item_id}", params={
        "api_key":              API_KEY,
        "append_to_response":  "credits,keywords",
    })
    if not data:
        return ""

    parts = []

    # Genres (high weight — repeat them so TF-IDF scores them higher)
    genres = [g["name"] for g in data.get("genres", [])]
    parts.extend(genres * 3)                       # triple weight

    # Overview
    overview = data.get("overview", "")
    if overview:
        parts.append(overview)

    # Keywords (very high signal)
    kw_key  = "keywords" if content_type == "movie" else "results"
    kw_data = data.get("keywords", {})
    keywords = [k["name"] for k in kw_data.get(kw_key if kw_key in kw_data else "keywords", [])]
    parts.extend(keywords * 2)                     # double weight

    # Cast — top 5 actors
    cast = [c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]
    parts.extend(cast)

    # Director (movies) or created_by (TV)
    if content_type == "movie":
        directors = [c["name"] for c in data.get("credits", {}).get("crew", []) if c.get("job") == "Director"]
        parts.extend(directors * 2)
    else:
        creators = [c["name"] for c in data.get("created_by", [])]
        parts.extend(creators * 2)

    return " ".join(parts).strip()

def enrich_dataframe(df):
    """Add rich tags to every row using parallel requests."""
    print(f"\n🔍 Enriching {len(df)} items with rich tags (parallel)...")
    rich_tags = {}

    items = list(zip(df["id"], df["type"]))
    total = len(items)

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as ex:
        futures = {ex.submit(build_rich_tags, iid, ctype): iid for iid, ctype in items}
        for i, future in enumerate(as_completed(futures), 1):
            item_id = futures[future]
            tags    = future.result()
            rich_tags[item_id] = tags if tags else df.loc[df["id"] == item_id, "tags"].values[0]
            if i % 500 == 0:
                print(f"   → {i}/{total} items enriched")

    df["tags"] = df["id"].map(rich_tags).fillna("")
    print(f"   ✓ All {total} items enriched")
    return df

# ── Step 3: Build and save ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  HIVEMIND Catalogue Builder")
    print("=" * 60)

    # Fetch raw catalogues in parallel (both content types simultaneously)
    with ThreadPoolExecutor(max_workers=2) as ex:
        movie_future = ex.submit(fetch_all_pages, "movie")
        tv_future    = ex.submit(fetch_all_pages, "tv")
        movies_df = movie_future.result()
        tv_df     = tv_future.result()

    # Combine
    master_df = pd.concat([movies_df, tv_df], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=["id"])
    print(f"\n📦 Combined catalogue: {len(master_df)} total items")

    # Enrich with rich tags
    master_df = enrich_dataframe(master_df)

    # Final quality filter — drop items with empty tags
    master_df = master_df[master_df["tags"].str.strip() != ""]
    print(f"   ✓ {len(master_df)} items after quality filter")

    # Save
    # We save ONLY the dataframe — app.py fits TF-IDF on startup.
    # This keeps the pickle small (< 50MB even at 20K items).
    pickle.dump(master_df.to_dict(), open("content_dict.pkl", "wb"))
    print(f"\n✅ Saved content_dict.pkl  ({len(master_df)} items)")
    print("   Now run:  python app.py")
    print("=" * 60)

if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱  Total time: {elapsed/60:.1f} minutes")
