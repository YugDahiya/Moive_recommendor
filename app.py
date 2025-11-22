import os
import logging
from flask import Flask, render_template, request, jsonify
import pickle
import requests
import pandas as pd

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Config ----
# Keep original hard-coded TMDB key if you prefer; but supporting env var is safer.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8").strip()
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "").strip()  # optional fallback
PLACEHOLDER_POSTER = "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"

app = Flask(__name__)

# ---- Utility HTTP GET with safe behavior ----
def safe_get(url, params=None, timeout=6):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r
    except requests.exceptions.RequestException as e:
        logger.debug("HTTP request failed: %s %s -> %s", url, params, e)
        return None

# ---- Fetch poster (TMDB) with graceful 404 handling ----
def fetch_poster(movie_id):
    """
    Fetches the movie poster URL from TMDB API.
    Returns a full URL string on success, or None on failure (including 404).
    """
    if not movie_id:
        logger.debug("fetch_poster: no movie_id provided.")
        return None

    base = "https://api.themoviedb.org/3"
    path = f"{base}/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    r = safe_get(path, params=params, timeout=6)
    if r is None:
        logger.info("fetch_poster: TMDB request failed for id %s (no response).", movie_id)
        return None

    if r.status_code == 404:
        # movie not found on TMDB
        logger.info("fetch_poster: TMDB returned 404 for movie id %s.", movie_id)
        return None

    if not r.ok:
        logger.warning("fetch_poster: TMDB returned status %s for id %s (body excerpt: %s)", r.status_code, movie_id, (r.text[:300] if r.text else ""))
        return None

    try:
        data = r.json()
    except Exception as e:
        logger.warning("fetch_poster: invalid JSON from TMDB for id %s: %s", movie_id, e)
        return None

    poster_path = data.get("poster_path")
    if poster_path:
        # Normalize poster path to full URL
        if poster_path.startswith("/"):
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            return poster_path

    logger.info("fetch_poster: no poster_path found on TMDB for id %s.", movie_id)
    return None

# ---- OMDb fallback (optional) ----
def omdb_get_by_title(title):
    """Try OMDb by title. Returns dict or None."""
    if not OMDB_API_KEY or not title:
        return None
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
    r = safe_get(url, params=params)
    if not r or not r.ok:
        return None
    try:
        data = r.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return None

def omdb_get_by_imdb_id(imdb_id):
    if not OMDB_API_KEY or not imdb_id:
        return None
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "i": imdb_id, "plot": "full"}
    r = safe_get(url, params=params)
    if not r or not r.ok:
        return None
    try:
        data = r.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return None

# ---- Load pickles (original style) ----
try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)

    # Ensure numeric columns
    if 'year' in movies.columns:
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    if 'vote_average' in movies.columns:
        movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce')

    if 'movie_id' not in movies.columns:
        movies['movie_id'] = None

    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
    logger.info("Loaded pickles successfully.")
except FileNotFoundError:
    logger.error("Model files not found. Please prepare 'artifacts/movie_dict.pkl' and 'artifacts/similarity.pkl'.")
    movies, similarity = None, None
except Exception as e:
    logger.exception("Error loading pickles: %s", e)
    movies, similarity = None, None

# ---- Helper: extract poster from row if present ----
def extract_existing_poster(row):
    for key in ('poster', 'poster_url', 'poster_path'):
        if key in row and isinstance(row.get(key), str) and row.get(key).strip():
            v = row.get(key).strip()
            if v.startswith("/"):
                return f"https://image.tmdb.org/t/p/w500{v}"
            return v
    return None

# ---- Format movie dict for template ----
def get_movie_details(movie_series):
    """
    Prefer poster already in dataset. Only call TMDB when missing.
    If TMDB fails, optionally try OMDb (if OMDB_API_KEY provided).
    Always return poster (placeholder if everything fails).
    """
    # Title
    title = movie_series.get('title') if 'title' in movie_series else movie_series.get('name') if 'name' in movie_series else "Untitled"

    # 1) prefer poster in the dataset
    poster_url = extract_existing_poster(movie_series)

    # 2) try TMDB if missing and movie_id present
    if not poster_url:
        movie_id = movie_series.get('movie_id')
        poster_from_tmdb = fetch_poster(movie_id)
        if poster_from_tmdb:
            poster_url = poster_from_tmdb

    # 3) try OMDb fallback if still missing and OMDB_API_KEY present
    if not poster_url and OMDB_API_KEY:
        # try IMDb id via TMDB first if movie_id exists
        imdb_id = None
        if movie_series.get('movie_id'):
            tm = safe_get(f"https://api.themoviedb.org/3/movie/{movie_series.get('movie_id')}", params={"api_key": TMDB_API_KEY})
            if tm and tm.ok:
                try:
                    js = tm.json()
                    imdb_id = js.get('imdb_id')
                except Exception:
                    imdb_id = None
        if imdb_id:
            om = omdb_get_by_imdb_id(imdb_id)
            if om and om.get('Poster') and om.get('Poster') != "N/A":
                poster_url = om.get('Poster')
        if not poster_url:
            om2 = omdb_get_by_title(title)
            if om2 and om2.get('Poster') and om2.get('Poster') != "N/A":
                poster_url = om2.get('Poster')

    # 4) fallback placeholder
    if not poster_url:
        poster_url = PLACEHOLDER_POSTER

    # rating & year
    try:
        rating_str = f"{movie_series.vote_average:.1f}"
    except Exception:
        rating_str = "N/A"

    return {
        'title': title,
        'poster': poster_url,
        'movie_id': movie_series.get('movie_id'),
        'year': int(movie_series.year) if pd.notna(movie_series.year) else "N/A",
        'rating': rating_str
    }

# ---- Recommendation ----
def recommend(movie):
    if movies is None or similarity is None or not movie:
        return []
    try:
        index = movies[movies['title'] == movie].index[0]
    except Exception:
        try:
            index = movies[movies['title'].str.lower() == str(movie).lower()].index[0]
        except Exception:
            return []
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distances[1:6]:
        movie_data = movies.iloc[i[0]]
        recommendations.append(get_movie_details(movie_data))
    return recommendations

# ---- Curated / Genre helpers (same as before) ----
def get_curated_movies(sort_by, n=20):
    if movies is None:
        return []
    if sort_by == 'vote_average':
        df_filtered = movies[(movies['vote_average'] <= 9.5) & (movies['vote_average'] > 0)]
    else:
        df_filtered = movies
    df_sorted = df_filtered.sort_values(sort_by, ascending=False).dropna(subset=[sort_by])
    curated_list = []
    for _, row in df_sorted.head(n).iterrows():
        curated_list.append(get_movie_details(row))
    return curated_list

def get_movies_by_genre(genre_name, n=20):
    if movies is None or 'genres' not in movies.columns:
        return []
    try:
        genre_movies = movies[movies['genres'].apply(lambda x: genre_name in x if isinstance(x, list) else False)]
    except TypeError:
        try:
            genre_movies = movies[movies['genres'].str.contains(genre_name, na=False)]
        except Exception:
            return []
    df_sorted = genre_movies.sort_values('vote_average', ascending=False)
    curated_list = []
    for _, row in df_sorted.head(n).iterrows():
        curated_list.append(get_movie_details(row))
    return curated_list

# ---- /movie_info route (safe TMDB + OMDb fallback) ----
@app.route('/movie_info', methods=['POST'])
def movie_info():
    if movies is None:
        return jsonify({"error": "movie dataset not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    movie_id = payload.get('movie_id')
    title = payload.get('title')

    # find row if possible
    row_info = None
    if movie_id:
        try:
            row_info = movies[movies['movie_id'] == movie_id].iloc[0]
        except Exception:
            row_info = None
    if row_info is None and title:
        try:
            row_info = movies[movies['title'].str.lower() == title.lower()].iloc[0]
        except Exception:
            row_info = None

    details = {}

    # Prefill from dataset if available
    if row_info is not None:
        details['title'] = row_info.get('title') or title
        if 'overview' in row_info and isinstance(row_info.get('overview'), str) and row_info.get('overview').strip():
            details['overview'] = row_info.get('overview')
        if 'vote_average' in row_info:
            try:
                details['rating'] = float(row_info.get('vote_average'))
            except Exception:
                details['rating'] = None

    # Try TMDB details if movie_id exists
    if movie_id:
        base = "https://api.themoviedb.org/3"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        r = safe_get(f"{base}/movie/{movie_id}", params=params, timeout=6)
        if r and r.ok:
            try:
                j = r.json()
                poster_path = j.get('poster_path')
                if poster_path:
                    details['poster'] = f"https://image.tmdb.org/t/p/w500{poster_path}"
                details['title'] = j.get('title') or details.get('title') or title
                details['overview'] = j.get('overview') or details.get('overview') or "No description available."
                details['rating'] = j.get('vote_average') or details.get('rating')
                details['release_date'] = j.get('release_date')
                details['runtime'] = j.get('runtime')
                details['genres'] = [g.get('name') for g in j.get('genres', []) if isinstance(g, dict)]
            except Exception as e:
                logger.debug("movie_info: failed parsing TMDB JSON for id %s: %s", movie_id, e)
        else:
            if r is not None and r.status_code == 404:
                logger.info("movie_info: TMDB returned 404 for id %s", movie_id)
            else:
                logger.info("movie_info: TMDB request failed for id %s (status: %s)", movie_id, getattr(r, "status_code", "no response"))

        # credits (cast)
        cred = safe_get(f"{base}/movie/{movie_id}/credits", params={"api_key": TMDB_API_KEY}, timeout=6)
        if cred and cred.ok:
            try:
                cj = cred.json()
                cast = cj.get('cast', [])[:6]
                top = []
                for c in cast:
                    top.append({
                        "name": c.get('name'),
                        "character": c.get('character'),
                        "profile": (f"https://image.tmdb.org/t/p/w185{c.get('profile_path')}" if c.get('profile_path') else None)
                    })
                details['cast'] = top
            except Exception:
                pass

    # If TMDB didn't produce poster/overview and OMDb key exists, try OMDb
    if OMDB_API_KEY and (('poster' not in details or details.get('poster') in (None, "", PLACEHOLDER_POSTER)) or ('overview' not in details or not details.get('overview'))):
        # try TMDB-imdb lookup if not done
        imdb_id = None
        if movie_id:
            r2 = safe_get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": TMDB_API_KEY})
            if r2 and r2.ok:
                try:
                    imdb_id = r2.json().get('imdb_id')
                except Exception:
                    imdb_id = None
        om = None
        if imdb_id:
            om = omdb_get_by_imdb_id(imdb_id)
        if not om and title:
            om = omdb_get_by_title(title)
        if om:
            if om.get('Poster') and om.get('Poster') != "N/A":
                details.setdefault('poster', om.get('Poster'))
            if om.get('Plot') and om.get('Plot') != "N/A":
                details.setdefault('overview', om.get('Plot'))
            if om.get('Actors') and om.get('Actors') != "N/A":
                actors = [a.strip() for a in om.get('Actors').split(",") if a.strip()]
                details.setdefault('cast', [{"name": a, "character": None, "profile": None} for a in actors[:6]])
            if om.get('Released'):
                details.setdefault('release_date', om.get('Released'))
            if om.get('Runtime'):
                try:
                    details.setdefault('runtime', int(str(om.get('Runtime')).split()[0]))
                except Exception:
                    details.setdefault('runtime', None)
            if om.get('Genre'):
                details.setdefault('genres', [g.strip() for g in om.get('Genre').split(",") if g.strip()])

    # Prefer poster from dataset if present
    if row_info is not None:
        p = extract_existing_poster(row_info)
        if p:
            details['poster'] = p

    details.setdefault('poster', PLACEHOLDER_POSTER)
    details.setdefault('title', title or details.get('title') or "Unknown")
    details.setdefault('overview', details.get('overview') or "No description available.")
    details.setdefault('rating', details.get('rating') or None)
    details.setdefault('cast', details.get('cast') or [])

    return jsonify(details)

# ---- Main route ----
@app.route('/', methods=['GET', 'POST'])
def index():
    if movies is None:
        return "<h3>Model files not found. Please make sure 'artifacts' folder is present.</h3>"

    movie_list = movies['title'].dropna().tolist() if 'title' in movies.columns else []

    recommendations = []
    selected_movie = None
    searched_movie_details = None
    top_rated = []
    action_movies = []
    comedy_movies = []
    drama_movies = []
    scifi_movies = []

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        recommendations = recommend(selected_movie)
        if selected_movie:
            try:
                movie_data = movies[movies['title'] == selected_movie].iloc[0]
                searched_movie_details = get_movie_details(movie_data)
            except Exception:
                try:
                    movie_data = movies[movies['title'].str.lower() == selected_movie.lower()].iloc[0]
                    searched_movie_details = get_movie_details(movie_data)
                except Exception:
                    logger.info("Searched movie '%s' not found.", selected_movie)
    else:
        top_rated = get_curated_movies('vote_average', 20)
        action_movies = get_movies_by_genre('Action', 20)
        comedy_movies = get_movies_by_genre('Comedy', 20)
        drama_movies = get_movies_by_genre('Drama', 20)
        scifi_movies = get_movies_by_genre('Science Fiction', 20)

    return render_template('index.html',
                           movie_list=movie_list,
                           recommendations=recommendations,
                           selected_movie=selected_movie,
                           searched_movie_details=searched_movie_details,
                           top_rated=top_rated,
                           action_movies=action_movies,
                           comedy_movies=comedy_movies,
                           drama_movies=drama_movies,
                           scifi_movies=scifi_movies)

if __name__ == '__main__':
    app.run(debug=True)
