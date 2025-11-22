import os
import pickle
import logging
import time
from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd

# ---------- Config & Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8").strip()
PLACEHOLDER_POSTER = "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"
BAD_IDS_PATH = "artifacts/bad_tmdb_ids.pkl"
TMDB_BASE = "https://api.themoviedb.org/3"

app = Flask(__name__)

# ---------- HTTP helper ----------
def safe_get(url, params=None, timeout=6):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r
    except requests.exceptions.RequestException as e:
        logger.debug("HTTP request failed: %s %s -> %s", url, params, e)
        return None

# ---------- blacklist ----------
def load_bad_ids():
    if os.path.exists(BAD_IDS_PATH):
        try:
            return set(pickle.load(open(BAD_IDS_PATH, "rb")))
        except Exception as e:
            logger.warning("Could not load bad ids file: %s", e)
    return set()

def save_bad_ids(bad_ids):
    try:
        os.makedirs(os.path.dirname(BAD_IDS_PATH), exist_ok=True)
        pickle.dump(list(bad_ids), open(BAD_IDS_PATH, "wb"))
    except Exception as e:
        logger.warning("Failed to save bad ids: %s", e)

bad_tmdb_ids = load_bad_ids()

# ---------- in-memory poster cache ----------
poster_cache = {}

def fetch_poster(movie_id):
    """Return poster URL or None. Uses blacklist and per-run cache."""
    if not movie_id:
        return None
    try:
        key = int(movie_id)
    except Exception:
        key = movie_id
    if key in poster_cache:
        return poster_cache[key]
    if key in bad_tmdb_ids:
        poster_cache[key] = None
        return None
    if not TMDB_API_KEY:
        poster_cache[key] = None
        return None

    url = f"{TMDB_BASE}/movie/{key}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = safe_get(url, params=params, timeout=5)
    if r is None:
        poster_cache[key] = None
        return None
    if r.status_code == 404:
        logger.info("fetch_poster: TMDB 404 for movie_id %s — blacklisting", key)
        bad_tmdb_ids.add(key)
        save_bad_ids(bad_tmdb_ids)
        poster_cache[key] = None
        return None
    if not r.ok:
        poster_cache[key] = None
        return None
    try:
        j = r.json()
        poster_path = j.get("poster_path")
        if poster_path:
            full = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path.startswith("/") else poster_path
            poster_cache[key] = full
            return full
    except Exception:
        pass
    poster_cache[key] = None
    return None

# ---------- load pickles ----------
try:
    movies_dict = pickle.load(open("artifacts/movie_dict.pkl","rb"))
    movies = pd.DataFrame(movies_dict)
    if "year" in movies.columns:
        movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    if "vote_average" in movies.columns:
        movies["vote_average"] = pd.to_numeric(movies["vote_average"], errors="coerce")
    if "movie_id" not in movies.columns:
        movies["movie_id"] = None
    similarity = pickle.load(open("artifacts/similarity.pkl","rb"))
    logger.info("Pickles loaded successfully.")
except FileNotFoundError:
    logger.error("Pickle files missing. Ensure artifacts/movie_dict.pkl and artifacts/similarity.pkl exist.")
    movies, similarity = None, None
except Exception as e:
    logger.exception("Error loading pickles: %s", e)
    movies, similarity = None, None

# ---------- helpers ----------
def extract_existing_poster(row):
    for key in ("poster","poster_url","poster_path"):
        if key in row and isinstance(row.get(key), str) and row.get(key).strip():
            v = row.get(key).strip()
            if v.startswith("/"):
                return f"https://image.tmdb.org/t/p/w500{v}"
            return v
    return None

def get_movie_details(movie_series):
    poster = extract_existing_poster(movie_series)
    if not poster:
        poster = fetch_poster(movie_series.get("movie_id"))
    if not poster:
        poster = PLACEHOLDER_POSTER
    try:
        rating_str = f"{movie_series.vote_average:.1f}"
    except Exception:
        rating_str = "N/A"
    return {
        "title": movie_series.get("title"),
        "poster": poster,
        "movie_id": movie_series.get("movie_id"),
        "year": int(movie_series.year) if pd.notna(movie_series.year) else "N/A",
        "rating": rating_str
    }

def recommend(movie):
    if movies is None or similarity is None or not movie:
        return []
    try:
        index = movies[movies["title"] == movie].index[0]
    except Exception:
        try:
            index = movies[movies["title"].str.lower() == str(movie).lower()].index[0]
        except Exception:
            return []
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recs=[]
    for pair in distances[1:6]:
        m = movies.iloc[pair[0]]
        recs.append(get_movie_details(m))
    return recs

def get_curated_movies(sort_by, n=20):
    if movies is None:
        return []
    df = movies
    if sort_by == "vote_average":
        df = movies[(movies["vote_average"] > 0) & (movies["vote_average"] <= 10)]
    df_sorted = df.dropna(subset=[sort_by]).sort_values(sort_by, ascending=False)
    out=[]
    for _, row in df_sorted.head(n).iterrows():
        out.append(get_movie_details(row))
    return out

def get_movies_by_genre(genre_name, n=20):
    if movies is None or "genres" not in movies.columns:
        return []
    try:
        genre_movies = movies[movies["genres"].apply(lambda x: genre_name in x if isinstance(x, list) else False)]
    except Exception:
        try:
            genre_movies = movies[movies["genres"].str.contains(genre_name, na=False)]
        except Exception:
            return []
    df_sorted = genre_movies.sort_values("vote_average", ascending=False)
    out=[]
    for _, row in df_sorted.head(n).iterrows():
        out.append(get_movie_details(row))
    return out

# ---------- TMDB live helpers ----------
def tmdb_search_title(query, page=1):
    if not TMDB_API_KEY or not query:
        return []
    params = {"api_key": TMDB_API_KEY, "query": query, "page": page, "include_adult": False, "language": "en-US"}
    r = safe_get(f"{TMDB_BASE}/search/movie", params=params)
    if not r or not r.ok:
        return []
    try:
        return r.json().get("results", [])
    except Exception:
        return []

def tmdb_get_recommendations(movie_id, n=6):
    if not TMDB_API_KEY or not movie_id:
        return []
    r = safe_get(f"{TMDB_BASE}/movie/{movie_id}/recommendations", params={"api_key": TMDB_API_KEY, "language":"en-US", "page":1})
    if not r or not r.ok:
        return []
    try:
        items = r.json().get("results", [])[:n]
        out=[]
        for it in items:
            out.append({
                "title": it.get("title"),
                "movie_id": it.get("id"),
                "poster": (f"https://image.tmdb.org/t/p/w500{it.get('poster_path')}" if it.get('poster_path') else None),
                "overview": it.get("overview")
            })
        return out
    except Exception:
        return []

# ---------- optional AJAX endpoint (client-side search) ----------
@app.route("/search_live", methods=["POST"])
def search_live():
    payload = request.get_json(silent=True) or {}
    q = payload.get("q", "").strip()
    if not q:
        return jsonify({"error":"no query"}), 400
    matches = tmdb_search_title(q, page=1)
    if not matches:
        return jsonify({"results": []})
    best = matches[0]
    recs = tmdb_get_recommendations(best.get("id"), n=6)
    result = {
        "match": {
            "title": best.get("title"),
            "movie_id": best.get("id"),
            "poster": (f"https://image.tmdb.org/t/p/w500{best.get('poster_path')}" if best.get('poster_path') else None),
            "overview": best.get("overview")
        },
        "tmdb_recs": recs
    }
    return jsonify(result)

# ---------- movie_info (existing) ----------
@app.route("/movie_info", methods=["POST"])
def movie_info():
    if movies is None:
        return jsonify({"error":"movies not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    movie_id = payload.get("movie_id")
    title = payload.get("title")
    region = payload.get("region") or "IN"

    row = None
    if movie_id:
        try:
            row = movies[movies["movie_id"] == movie_id].iloc[0]
        except Exception:
            row = None
    if row is None and title:
        try:
            row = movies[movies["title"].str.lower() == title.lower()].iloc[0]
        except Exception:
            row = None

    details = {}
    if row is not None:
        details["title"] = row.get("title") or title
        if "overview" in row and isinstance(row.get("overview"), str) and row.get("overview").strip():
            details["overview"] = row.get("overview")
        if "vote_average" in row:
            try:
                details["rating"] = float(row.get("vote_average"))
            except Exception:
                details["rating"] = None

    # TMDB calls only when movie_id present & not blacklisted
    if movie_id and movie_id not in bad_tmdb_ids and TMDB_API_KEY:
        base = TMDB_BASE
        params_base = {"api_key": TMDB_API_KEY, "language": "en-US"}

        # /movie/{id}
        r = safe_get(f"{base}/movie/{movie_id}", params=params_base, timeout=5)
        if r and r.ok:
            try:
                j = r.json()
                poster_path = j.get("poster_path")
                if poster_path:
                    details["poster"] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                details["title"] = j.get("title") or details.get("title") or title
                details["overview"] = j.get("overview") or details.get("overview") or "No description available."
                details["rating"] = j.get("vote_average") or details.get("rating")
                details["release_date"] = j.get("release_date")
                details["runtime"] = j.get("runtime")
                details["genres"] = [g.get("name") for g in j.get("genres", []) if isinstance(g, dict)]
            except Exception:
                pass
        else:
            if r is not None and r.status_code == 404:
                bad_tmdb_ids.add(movie_id)
                save_bad_ids(bad_tmdb_ids)
            else:
                logger.debug("movie_info: TMDB movie request failed for id %s", movie_id)

        # /credits
        cred = safe_get(f"{base}/movie/{movie_id}/credits", params={"api_key": TMDB_API_KEY}, timeout=5)
        if cred and cred.ok:
            try:
                cj = cred.json()
                cast = cj.get("cast", [])[:12]
                top = []
                for c in cast:
                    top.append({
                        "id": c.get("id"),
                        "name": c.get("name"),
                        "character": c.get("character"),
                        "profile": (f"https://image.tmdb.org/t/p/w185{c.get('profile_path')}" if c.get("profile_path") else None)
                    })
                details["cast"] = top
            except Exception:
                details.setdefault("cast", [])

        # /videos
        vids = safe_get(f"{base}/movie/{movie_id}/videos", params={"api_key": TMDB_API_KEY, "language": "en-US"}, timeout=5)
        if vids and vids.ok:
            try:
                vj = vids.json()
                results = vj.get("results", [])
                details["videos"] = [{"id": v.get("id"), "name": v.get("name"), "site": v.get("site"), "type": v.get("type"), "key": v.get("key"), "official": v.get("official", False)} for v in results if v.get("key")]
            except Exception:
                details.setdefault("videos", [])
        else:
            details.setdefault("videos", [])

        # /watch/providers
        wp = safe_get(f"{base}/movie/{movie_id}/watch/providers", params={"api_key": TMDB_API_KEY}, timeout=5)
        providers_list = []
        if wp and wp.ok:
            try:
                wj = wp.json()
                results = wj.get("results", {}) or {}
                chosen_country = region if region in results else (next(iter(results), None))
                if chosen_country:
                    country_info = results.get(chosen_country, {})
                    for typ in ("flatrate","rent","buy"):
                        items = country_info.get(typ) or []
                        for item in items:
                            providers_list.append({
                                "provider_name": item.get("provider_name"),
                                "logo_path": (f"https://image.tmdb.org/t/p/w92{item.get('logo_path')}" if item.get("logo_path") else None),
                                "display_priority": item.get("display_priority"),
                                "type": typ
                            })
            except Exception:
                pass
        details["providers"] = providers_list

    # fallback poster logic
    if "poster" not in details or not details.get("poster"):
        if row is not None:
            p = extract_existing_poster(row)
            if p:
                details["poster"] = p
        if "poster" not in details or not details.get("poster"):
            p2 = fetch_poster(movie_id) if movie_id else None
            if p2:
                details["poster"] = p2

    details.setdefault("poster", PLACEHOLDER_POSTER)
    details.setdefault("title", title or details.get("title") or "Unknown")
    details.setdefault("overview", details.get("overview") or "No description available.")
    details.setdefault("rating", details.get("rating") or None)
    details.setdefault("cast", details.get("cast") or [])
    details.setdefault("videos", details.get("videos") or [])
    details.setdefault("providers", details.get("providers") or [])

    return jsonify(details)

# ---------- index route with TMDB fallback ----------
@app.route("/", methods=["GET","POST"])
def index():
    if movies is None:
        return "<h3>Model files not found. Please ensure artifacts/movie_dict.pkl and artifacts/similarity.pkl exist.</h3>"

    movie_list = movies["title"].dropna().tolist() if "title" in movies.columns else []

    recommendations=[]
    selected_movie=None
    searched_movie_details=None
    top_rated=[]
    action_movies=[]
    comedy_movies=[]
    drama_movies=[]
    scifi_movies=[]
    from_tmdb = False  # flag for UI

    if request.method == "POST":
        selected_movie = request.form.get("movie")
        if selected_movie:
            # If movie exists in local dataset -> use local recommend()
            if selected_movie in movie_list:
                recommendations = recommend(selected_movie)
                try:
                    movie_data = movies[movies["title"] == selected_movie].iloc[0]
                    searched_movie_details = get_movie_details(movie_data)
                except Exception:
                    searched_movie_details = None
            else:
                # fallback: search TMDB live
                logger.info("Selected movie not found locally — falling back to TMDB search for '%s'", selected_movie)
                tmdb_matches = tmdb_search_title(selected_movie, page=1)
                if tmdb_matches:
                    best = tmdb_matches[0]
                    tmdb_id = best.get("id")
                    searched_movie_details = {
                        "title": best.get("title"),
                        "poster": (f"https://image.tmdb.org/t/p/w500{best.get('poster_path')}" if best.get('poster_path') else PLACEHOLDER_POSTER),
                        "movie_id": tmdb_id,
                        "year": (best.get("release_date")[:4] if best.get("release_date") else None),
                        "rating": best.get("vote_average")
                    }
                    # get TMDB-based recommendations
                    recommendations = tmdb_get_recommendations(tmdb_id, n=5)
                    from_tmdb = True
                    # small sleep to be polite with TMDB (avoid leaky loops)
                    time.sleep(0.12)
                else:
                    # no TMDB match — leave empty
                    recommendations = []
                    searched_movie_details = None
    else:
        # landing page content (local curated)
        top_rated = get_curated_movies("vote_average", 20)
        action_movies = get_movies_by_genre("Action", 20)
        comedy_movies = get_movies_by_genre("Comedy", 20)
        drama_movies = get_movies_by_genre("Drama", 20)
        scifi_movies = get_movies_by_genre("Science Fiction", 20)

    return render_template("index.html",
                           movie_list=movie_list,
                           recommendations=recommendations,
                           selected_movie=selected_movie,
                           searched_movie_details=searched_movie_details,
                           top_rated=top_rated,
                           action_movies=action_movies,
                           comedy_movies=comedy_movies,
                           drama_movies=drama_movies,
                           scifi_movies=scifi_movies,
                           from_tmdb=from_tmdb)

if __name__ == "__main__":
    app.run(debug=True)
