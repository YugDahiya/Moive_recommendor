import os
import pickle
import logging
from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd

# ---------- Config & Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Use your existing key here (kept same as earlier). You may set TMDB_API_KEY env var to override.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8").strip()
PLACEHOLDER_POSTER = "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"
BAD_IDS_PATH = "artifacts/bad_tmdb_ids.pkl"

app = Flask(__name__)

# ---------- Simple safe GET ----------
def safe_get(url, params=None, timeout=5):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r
    except requests.exceptions.RequestException as e:
        logger.debug("HTTP request failed: %s %s -> %s", url, params, e)
        return None

# ---------- Load blacklist of bad ids (persistent) ----------
def load_bad_ids():
    if os.path.exists(BAD_IDS_PATH):
        try:
            return set(pickle.load(open(BAD_IDS_PATH, "rb")))
        except Exception as e:
            logger.warning("Failed to load bad ids file: %s", e)
    return set()

def save_bad_ids(bad_ids):
    try:
        os.makedirs(os.path.dirname(BAD_IDS_PATH), exist_ok=True)
        pickle.dump(list(bad_ids), open(BAD_IDS_PATH, "wb"))
    except Exception as e:
        logger.warning("Failed to save bad ids: %s", e)

bad_tmdb_ids = load_bad_ids()

# ---------- In-memory poster cache (per-run) ----------
poster_cache = {}

# ---------- Fetch poster: prefer dataset, avoid blacklisted ids ----------
def fetch_poster(movie_id):
    """
    Returns poster URL or None.
    Logic:
      - If movie_id in poster_cache -> return cached value (may be None)
      - If movie_id in bad_tmdb_ids -> return None immediately
      - Call TMDB /movie/{id} with timeout; if 404 -> add to bad list and return None
      - If poster_path returned -> build full URL and cache/return
      - On other failures return None
    """
    if not movie_id:
        return None

    # cached (including negative cache)
    if movie_id in poster_cache:
        return poster_cache[movie_id]

    # blocked id
    if movie_id in bad_tmdb_ids:
        logger.debug("fetch_poster: movie_id %s is blacklisted, skipping TMDB fetch", movie_id)
        poster_cache[movie_id] = None
        return None

    if not TMDB_API_KEY:
        logger.debug("fetch_poster: no TMDB_API_KEY set")
        poster_cache[movie_id] = None
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = safe_get(url, params=params, timeout=5)
    if r is None:
        poster_cache[movie_id] = None
        return None

    if r.status_code == 404:
        logger.info("fetch_poster: TMDB 404 for movie_id %s — adding to blacklist", movie_id)
        bad_tmdb_ids.add(movie_id)
        save_bad_ids(bad_tmdb_ids)
        poster_cache[movie_id] = None
        return None

    if not r.ok:
        logger.info("fetch_poster: TMDB returned %s for id %s", r.status_code, movie_id)
        poster_cache[movie_id] = None
        return None

    try:
        j = r.json()
    except Exception:
        poster_cache[movie_id] = None
        return None

    poster_path = j.get("poster_path")
    if poster_path:
        # normalize
        if poster_path.startswith("/"):
            full = f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            full = poster_path
        poster_cache[movie_id] = full
        return full

    poster_cache[movie_id] = None
    return None

# ---------- Load pickles ----------
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
    logger.info("Loaded pickles OK.")
except FileNotFoundError:
    logger.error("Pickle files not found. Provide artifacts/movie_dict.pkl and artifacts/similarity.pkl")
    movies, similarity = None, None
except Exception as e:
    logger.exception("Error loading pickles: %s", e)
    movies, similarity = None, None

# ---------- Helpers ----------
def extract_existing_poster(row):
    for key in ("poster","poster_url","poster_path"):
        if key in row and isinstance(row.get(key), str) and row.get(key).strip():
            v = row.get(key).strip()
            if v.startswith("/"):
                return f"https://image.tmdb.org/t/p/w500{v}"
            return v
    return None

def get_movie_details(movie_series):
    # prefer dataset poster
    poster = extract_existing_poster(movie_series)
    if not poster:
        # try tmdb (blocked ids are skipped inside fetch_poster)
        poster = fetch_poster(movie_series.get("movie_id"))

    if not poster:
        poster = "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"

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

# ---------- Endpoints ----------
@app.route("/movie_info", methods=["POST"])
def movie_info():
    """
    Returns JSON with poster, title, overview, rating, release_date, runtime, genres and (optionally) cast.
    Poster retrieval uses the same lightweight logic and will NOT attempt TMDB for blacklisted ids.
    """
    if movies is None:
        return jsonify({"error":"movies not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    movie_id = payload.get("movie_id")
    title = payload.get("title")

    # Try to find row in DF if possible
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
    # Prefer dataset values
    if row is not None:
        details["title"] = row.get("title") or title
        if "overview" in row and isinstance(row.get("overview"), str) and row.get("overview").strip():
            details["overview"] = row.get("overview")
        if "vote_average" in row:
            try:
                details["rating"] = float(row.get("vote_average"))
            except Exception:
                details["rating"] = None

    # minimal TMDB attempt for details ONLY if we have movie_id and not blacklisted
    if movie_id and movie_id not in bad_tmdb_ids and TMDB_API_KEY:
        r = safe_get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": TMDB_API_KEY, "language":"en-US"}, timeout=5)
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
                logger.info("movie_info: TMDB 404 for id %s; adding to blacklist", movie_id)
                bad_tmdb_ids.add(movie_id)
                save_bad_ids(bad_tmdb_ids)
            else:
                logger.debug("movie_info: TMDB request failed for id %s (status: %s)", movie_id, getattr(r,"status_code",None))

    # If poster still missing, try extract from dataset OR cached fetch_poster (which respects blacklist)
    if "poster" not in details or not details.get("poster"):
        if row is not None:
            p = extract_existing_poster(row)
            if p:
                details["poster"] = p
        if "poster" not in details or not details.get("poster"):
            # try fetch_poster (will skip blacklisted ids)
            p2 = fetch_poster(movie_id) if movie_id else None
            if p2:
                details["poster"] = p2

    # final fallback
    details.setdefault("poster", PLACEHOLDER_POSTER)
    details.setdefault("title", title or details.get("title") or "Unknown")
    details.setdefault("overview", details.get("overview") or "No description available.")
    details.setdefault("rating", details.get("rating") or None)
    details.setdefault("cast", details.get("cast") or [])

    return jsonify(details)

@app.route("/", methods=["GET","POST"])
def index():
    if movies is None:
        return "<h3>Model files not found. Please ensure artifacts/movie_dict.pkl and similarity.pkl exist.</h3>"

    movie_list = movies["title"].dropna().tolist() if "title" in movies.columns else []

    recommendations = []
    selected_movie = None
    searched_movie_details = None
    top_rated = []
    action_movies = []
    comedy_movies = []
    drama_movies = []
    scifi_movies = []

    if request.method == "POST":
        selected_movie = request.form.get("movie")
        recommendations = recommend(selected_movie)
        if selected_movie:
            try:
                movie_data = movies[movies["title"] == selected_movie].iloc[0]
                searched_movie_details = get_movie_details(movie_data)
            except Exception:
                try:
                    movie_data = movies[movies["title"].str.lower() == selected_movie.lower()].iloc[0]
                    searched_movie_details = get_movie_details(movie_data)
                except Exception:
                    logger.info("Searched movie not found: %s", selected_movie)
    else:
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
                           scifi_movies=scifi_movies)

if __name__ == "__main__":
    app.run(debug=True)
