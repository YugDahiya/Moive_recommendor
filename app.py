import os, pickle, logging, re, time, threading
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TMDB_API_KEY        = "8265bd1679663a7ea12ac168da84d2e8"
PLACEHOLDER_POSTER  = "https://placehold.co/500x750/0d1520/6eaee8?text=No+Poster"
TMDB_BASE           = "https://api.themoviedb.org/3"
MAX_RECOMMENDATIONS = 20
MAX_INFINITE_PAGES  = 10

# ── In-memory cache ───────────────────────────────────────────────────────────
# Stores (data, expiry_timestamp). TTL in seconds.
_cache      = {}
_cache_lock = threading.Lock()
CAROUSEL_TTL  = 3600   # home carousels refresh every 1 hour
GENRE_TTL     = 1800   # genre lists refresh every 30 min
SEARCH_TTL    = 600    # search results cache 10 min

def cache_get(key):
    with _cache_lock:
        item = _cache.get(key)
        if item and time.time() < item[1]:
            return item[0]
        return None

def cache_set(key, value, ttl):
    with _cache_lock:
        _cache[key] = (value, time.time() + ttl)

GENRES = ["Action","Adventure","Animation","Comedy","Crime","Documentary",
          "Drama","Fantasy","Horror","Mystery","Romance","Science Fiction",
          "Thriller","Western","Family"]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hivemind-dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///hivemind.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ── Connection pool for requests ──────────────────────────────────────────────
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=20,
    pool_maxsize=40,
    max_retries=requests.adapters.Retry(total=2, backoff_factor=0.2)
)
_session.mount("https://", _adapter)

db = SQLAlchemy(app)

# ── Models ────────────────────────────────────────────────────────────────────
class User(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(120), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    library       = db.relationship("LibraryItem", backref="user", lazy=True, cascade="all, delete-orphan")
    history       = db.relationship("HistoryItem",  backref="user", lazy=True, cascade="all, delete-orphan")

class LibraryItem(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tmdb_id    = db.Column(db.Integer, nullable=False)
    media_type = db.Column(db.String(10), default="movie")
    title      = db.Column(db.String(300))
    poster     = db.Column(db.String(500))
    rating     = db.Column(db.Float)
    added_at   = db.Column(db.DateTime, default=datetime.utcnow)

class HistoryItem(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    tmdb_id    = db.Column(db.Integer, nullable=False)
    media_type = db.Column(db.String(10), default="movie")
    title      = db.Column(db.String(300))
    poster     = db.Column(db.String(500))
    rating     = db.Column(db.Float)
    viewed_at  = db.Column(db.DateTime, default=datetime.utcnow)

# ── ML Loading ────────────────────────────────────────────────────────────────
master_df = tfidf = tfidf_matrix = None
for _p in ["content_dict.pkl", "artifacts/content_dict.pkl"]:
    if os.path.exists(_p):
        try:
            with open(_p, "rb") as f:
                master_df = pd.DataFrame(pickle.load(f))
            master_df["tags"]   = master_df["tags"].fillna("")
            master_df["poster"] = master_df.get("poster", pd.Series([""] * len(master_df))).fillna("")
            tfidf        = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(master_df["tags"])
            logger.info(f"✓ Loaded {len(master_df)} items from {_p}")
            break
        except Exception as e:
            logger.error(f"Could not load {_p}: {e}")

# ── HTTP helper (uses persistent session + connection pool) ───────────────────
def safe_get(url, params=None, timeout=7):
    try:
        r = _session.get(url, params=params, timeout=timeout)
        return r if r.ok else None
    except Exception:
        return None

# ── Auth helpers ──────────────────────────────────────────────────────────────
def current_user():
    uid = session.get("user_id")
    return User.query.get(uid) if uid else None

def login_required(f):
    @wraps(f)
    def dec(*a, **kw):
        if not session.get("user_id"):
            flash("Please sign in.", "info")
            return redirect(url_for("login_page"))
        return f(*a, **kw)
    return dec

# ── TMDB helpers ──────────────────────────────────────────────────────────────
def tmdb_to_card(item, media_type=None):
    mt = media_type or item.get("media_type", "movie")
    pp = item.get("poster_path")
    return {
        "id":     item.get("id"),
        "title":  item.get("title") or item.get("name", "Unknown"),
        "poster": f"https://image.tmdb.org/t/p/w300{pp}" if pp else PLACEHOLDER_POSTER,
        "rating": round(item.get("vote_average", 0), 1),
        "type":   mt,
    }

def _resolve_genre_ids(genre_name):
    """Cached genre ID lookup — avoids hitting TMDB on every genre page load."""
    cache_key = f"genre_ids:{genre_name.lower()}"
    cached = cache_get(cache_key)
    if cached: return cached

    mid = tvid = None
    for ct in ("movie", "tv"):
        ck = f"genre_list:{ct}"
        genre_list = cache_get(ck)
        if not genre_list:
            r = safe_get(f"{TMDB_BASE}/genre/{ct}/list", params={"api_key": TMDB_API_KEY})
            genre_list = r.json().get("genres", []) if r else []
            cache_set(ck, genre_list, GENRE_TTL)
        for gg in genre_list:
            if gg["name"].lower() == genre_name.lower():
                if ct == "movie": mid  = gg["id"]
                else:             tvid = gg["id"]
                break

    result = (mid, tvid)
    cache_set(cache_key, result, GENRE_TTL)
    return result

# ── Home carousel fetcher (cached + parallel) ─────────────────────────────────
_CAROUSEL_ENDPOINTS = {
    "trending_movies": (f"{TMDB_BASE}/trending/movie/week", "movie"),
    "trending_tv":     (f"{TMDB_BASE}/trending/tv/week",    "tv"),
    "popular_movies":  (f"{TMDB_BASE}/movie/popular",       "movie"),
    "popular_tv":      (f"{TMDB_BASE}/tv/popular",          "tv"),
    "top_movies":      (f"{TMDB_BASE}/movie/top_rated",     "movie"),
    "top_tv":          (f"{TMDB_BASE}/tv/top_rated",        "tv"),
}

def _fetch_one_carousel(key, url, mt):
    cached = cache_get(f"carousel:{key}")
    if cached: return key, cached
    r = safe_get(url, params={"api_key": TMDB_API_KEY})
    data = [tmdb_to_card(i, mt) for i in (r.json().get("results", [])[:20] if r else [])]
    cache_set(f"carousel:{key}", data, CAROUSEL_TTL)
    return key, data

def fetch_home_carousels():
    results = {}
    # Only fetch what isn't cached yet — fully cached hits return immediately
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_fetch_one_carousel, k, u, mt): k
                for k, (u, mt) in _CAROUSEL_ENDPOINTS.items()}
        for fut in as_completed(futs):
            k, data = fut.result()
            results[k] = data
    return results

# Pre-warm carousels in background on startup so first request is instant
def _prewarm():
    try:
        logger.info("Pre-warming carousel cache…")
        fetch_home_carousels()
        logger.info("Carousel cache ready.")
    except Exception as e:
        logger.warning(f"Pre-warm failed: {e}")

threading.Thread(target=_prewarm, daemon=True).start()

# ── Context processor ─────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    user    = current_user()
    lib_ids = {i.tmdb_id for i in user.library} if user else set()
    return dict(current_user=user, genres=GENRES, library_ids=lib_ids)

# ── ML Engine ─────────────────────────────────────────────────────────────────
def build_rich_query_tags(item_id, m_type):
    """Build rich tag string matching the catalogue schema for accurate similarity."""
    r = safe_get(f"{TMDB_BASE}/{m_type}/{item_id}", params={
        "api_key": TMDB_API_KEY, "append_to_response": "credits,keywords"})
    if not r: return ""
    d = r.json(); parts = []
    genres = [g["name"] for g in d.get("genres", [])]
    parts.extend(genres * 3)
    if d.get("overview"): parts.append(d["overview"])
    kw = d.get("keywords", {})
    parts.extend([k["name"] for k in kw.get("keywords", kw.get("results", []))] * 2)
    parts.extend([c["name"] for c in d.get("credits", {}).get("cast", [])[:5]])
    if m_type == "movie":
        parts.extend([c["name"] for c in d.get("credits", {}).get("crew", []) if c.get("job") == "Director"] * 2)
    else:
        parts.extend([c["name"] for c in d.get("created_by", [])] * 2)
    return " ".join(parts).strip()

def get_tmdb_taste_profile(query):
    cache_key = f"taste:{query.lower().strip()}"
    cached = cache_get(cache_key)
    if cached: return cached

    r = safe_get(f"{TMDB_BASE}/search/multi", params={"api_key": TMDB_API_KEY, "query": query})
    if not r: return None, None
    results = r.json().get("results", [])
    if not results: return None, None

    # Skip persons — only use movie or tv results
    best = next((x for x in results if x.get("media_type") in ("movie", "tv")), None)
    if not best: return None, None

    m_type  = best["media_type"]
    item_id = best["id"]
    dr = safe_get(f"{TMDB_BASE}/{m_type}/{item_id}", params={"api_key": TMDB_API_KEY})
    if not dr: return None, None
    d = dr.json()
    taste_text = build_rich_query_tags(item_id, m_type)
    if not taste_text:
        taste_text = " ".join(g["name"] for g in d.get("genres", [])) + " " + d.get("overview","")
    ui = {
        "title":  d.get("title") or d.get("name"),
        "poster": f"https://image.tmdb.org/t/p/w300{d['poster_path']}" if d.get("poster_path") else PLACEHOLDER_POSTER,
        "rating": round(d.get("vote_average", 0), 1),
        "id": item_id, "type": m_type,
    }
    result = (taste_text, ui)
    cache_set(cache_key, result, SEARCH_TTL)
    return result

def _fetch_poster(item_id, media_type):
    """Fetch poster from TMDB with 24h cache."""
    key = f"poster:{item_id}"
    cached = cache_get(key)
    if cached: return cached
    r = safe_get(f"{TMDB_BASE}/{media_type}/{item_id}", params={"api_key": TMDB_API_KEY})
    if r:
        pp = r.json().get("poster_path")
        if pp:
            url = f"https://image.tmdb.org/t/p/w300{pp}"
            cache_set(key, url, 86400)
            return url
    return PLACEHOLDER_POSTER

def get_recommendations(taste_text):
    if master_df is None: return [], []
    uv     = tfidf.transform([taste_text])
    scores = cosine_similarity(uv, tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1]
    out_m, out_tv = [], []

    # Collect candidates separately per type so each bucket always fills to MAX
    movies_cands, tv_cands = [], []
    for i in idx:
        if scores[i] >= 0.98: continue
        if len(movies_cands) >= MAX_RECOMMENDATIONS and len(tv_cands) >= MAX_RECOMMENDATIONS: break
        row = master_df.iloc[i]
        bucket = movies_cands if row["type"] == "movie" else tv_cands
        if len(bucket) >= MAX_RECOMMENDATIONS: continue
        stored = str(row["poster"]) if "poster" in master_df.columns and row.get("poster") else ""
        bucket.append({
            "title":  row["title"],
            "rating": round(float(row["rating"]), 1),
            "id":     int(row["id"]),
            "type":   row["type"],
            "poster": stored,
        })
    candidates = movies_cands + tv_cands

    # Parallel-fetch posters for items that dont have a stored URL
    need = [c for c in candidates if not c["poster"]]
    if need:
        with ThreadPoolExecutor(max_workers=12) as ex:
            fmap = {ex.submit(_fetch_poster, c["id"], c["type"]): c for c in need}
            for fut, c in fmap.items():
                c["poster"] = fut.result()

    for c in candidates:
        if not c["poster"]: c["poster"] = PLACEHOLDER_POSTER
        if   c["type"] == "movie" and len(out_m)  < MAX_RECOMMENDATIONS: out_m.append(c)
        elif c["type"] == "tv"    and len(out_tv) < MAX_RECOMMENDATIONS: out_tv.append(c)
        if len(out_m) >= MAX_RECOMMENDATIONS and len(out_tv) >= MAX_RECOMMENDATIONS: break

    return out_m, out_tv

# ═════════════════════════════════════════════════════════════════════════════
# AUTH
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/signup", methods=["GET","POST"])
def signup_page():
    if session.get("user_id"): return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        if not name or not email or not pw: error = "All fields are required."
        elif len(pw) < 6:                   error = "Password must be at least 6 characters."
        elif User.query.filter_by(email=email).first(): error = "Email already exists."
        else:
            u = User(name=name, email=email, password_hash=generate_password_hash(pw))
            db.session.add(u); db.session.commit(); session["user_id"] = u.id
            return redirect(url_for("index"))
    return render_template("auth.html", mode="signup", error=error)

@app.route("/login", methods=["GET","POST"])
def login_page():
    if session.get("user_id"): return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pw    = request.form.get("password","")
        u = User.query.filter_by(email=email).first()
        if u and check_password_hash(u.password_hash, pw):
            session["user_id"] = u.id; return redirect(url_for("index"))
        error = "Invalid email or password."
    return render_template("auth.html", mode="login", error=error)

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("index"))

# ═════════════════════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET","POST"])
def index():
    search_data = movies_rec = series_rec = None
    query = None
    carousels = fetch_home_carousels()
    if request.method == "POST":
        query = request.form.get("movie","").strip()
        taste_text, search_data = get_tmdb_taste_profile(query)
        if taste_text:
            movies_rec, series_rec = get_recommendations(taste_text)
    return render_template("index.html",
        searched_movie_details=search_data,
        recommended_movies=movies_rec or [],
        recommended_series=series_rec or [],
        selected_movie=query,
        active_page="home", page_title=None, page_items=None,
        active_genre=None, infinite_type=None, infinite_genre=None,
        carousels=carousels)

# ═════════════════════════════════════════════════════════════════════════════
# TRENDING
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/trending")
def trending():
    cached = cache_get("trending_page")
    if not cached:
        r = safe_get(f"{TMDB_BASE}/trending/all/week", params={"api_key": TMDB_API_KEY, "page": 1})
        cached = [tmdb_to_card(i) for i in (r.json().get("results",[]) if r else []) if i.get("media_type") in ("movie","tv")]
        cache_set("trending_page", cached, CAROUSEL_TTL)
    return render_template("index.html", page_title="Trending This Week", page_items=cached,
        active_page="trending", active_genre=None, infinite_type="trending", infinite_genre=None,
        searched_movie_details=None, recommended_movies=[], recommended_series=[],
        selected_movie=None, carousels={})

@app.route("/api/trending")
def api_trending():
    page = request.args.get("page", 2, type=int)
    cached = cache_get(f"trending_api:{page}")
    if cached: return jsonify(cached)
    r = safe_get(f"{TMDB_BASE}/trending/all/week", params={"api_key": TMDB_API_KEY, "page": page})
    items = []; has_more = False
    if r:
        data = r.json()
        items    = [tmdb_to_card(i) for i in data.get("results",[]) if i.get("media_type") in ("movie","tv")]
        has_more = page < min(data.get("total_pages",1), MAX_INFINITE_PAGES)
    result = {"items": items, "has_more": has_more, "page": page}
    cache_set(f"trending_api:{page}", result, CAROUSEL_TTL)
    return jsonify(result)

# ═════════════════════════════════════════════════════════════════════════════
# GENRE
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/genre/<genre_name>")
def genre_page(genre_name):
    cache_key = f"genre_page:{genre_name.lower()}"
    cached = cache_get(cache_key)
    if cached is None:
        mid, tvid = _resolve_genre_ids(genre_name)
        items = []
        def _fetch_genre(ct, gid):
            try:
                r = safe_get(f"{TMDB_BASE}/discover/{ct}", params={
                    "api_key": TMDB_API_KEY, "with_genres": gid,
                    "sort_by": "popularity.desc", "page": 1})
                if not r: return []
                return [tmdb_to_card(i, ct) for i in r.json().get("results", [])]
            except Exception:
                return []
        futs = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            if mid:  futs.append(ex.submit(_fetch_genre, "movie", mid))
            if tvid: futs.append(ex.submit(_fetch_genre, "tv",    tvid))
        for fut in futs:
            try:
                result = fut.result()
                if isinstance(result, list):
                    items.extend(result)
            except Exception:
                pass
        cache_set(cache_key, items, GENRE_TTL)
        cached = items
    return render_template("index.html", page_title=genre_name, page_items=cached,
        active_page="genres", active_genre=genre_name, infinite_type="genre", infinite_genre=genre_name,
        searched_movie_details=None, recommended_movies=[], recommended_series=[],
        selected_movie=None, carousels={})

@app.route("/api/genre/<genre_name>")
def api_genre(genre_name):
    page = request.args.get("page", 2, type=int)
    cache_key = f"genre_api:{genre_name.lower()}:{page}"
    cached = cache_get(cache_key)
    if cached is not None: return jsonify(cached)
    mid, tvid = _resolve_genre_ids(genre_name)
    items = []; has_more = False
    def _fetch_genre_page(ct, gid):
        try:
            r = safe_get(f"{TMDB_BASE}/discover/{ct}", params={
                "api_key": TMDB_API_KEY, "with_genres": gid,
                "sort_by": "popularity.desc", "page": page})
            if not r: return [], False
            data = r.json()
            cards   = [tmdb_to_card(i, ct) for i in data.get("results", [])]
            more    = page < min(data.get("total_pages", 1), MAX_INFINITE_PAGES)
            return cards, more
        except Exception:
            return [], False
    futs = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        if mid:  futs["movie"] = ex.submit(_fetch_genre_page, "movie", mid)
        if tvid: futs["tv"]    = ex.submit(_fetch_genre_page, "tv",    tvid)
    for fut in futs.values():
        try:
            cards, more = fut.result()
            if isinstance(cards, list):
                items.extend(cards)
            has_more = has_more or bool(more)
        except Exception:
            pass
    result = {"items": items, "has_more": has_more, "page": page}
    cache_set(cache_key, result, GENRE_TTL)
    return jsonify(result)

# ═════════════════════════════════════════════════════════════════════════════
# LIBRARY
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/library")
@login_required
def library_page():
    u = current_user()
    items = [{"id": l.tmdb_id, "title": l.title, "poster": l.poster, "rating": l.rating, "type": l.media_type} for l in reversed(u.library)]
    return render_template("index.html", page_title="My Library", page_items=items,
        active_page="library", active_genre=None, infinite_type=None, infinite_genre=None,
        searched_movie_details=None, recommended_movies=[], recommended_series=[],
        selected_movie=None, carousels={})

@app.route("/library/add", methods=["POST"])
@login_required
def library_add():
    data = request.get_json(); u = current_user()
    tid  = int(data.get("id", 0))
    if not tid: return jsonify({"error": "No id"}), 400
    if LibraryItem.query.filter_by(user_id=u.id, tmdb_id=tid).first():
        return jsonify({"status": "already_added"})
    db.session.add(LibraryItem(user_id=u.id, tmdb_id=tid, media_type=data.get("type","movie"),
        title=data.get("title",""), poster=data.get("poster",""), rating=float(data.get("rating",0) or 0)))
    db.session.commit(); return jsonify({"status": "added"})

@app.route("/library/remove", methods=["POST"])
@login_required
def library_remove():
    data = request.get_json(); u = current_user()
    item = LibraryItem.query.filter_by(user_id=u.id, tmdb_id=int(data.get("id",0))).first()
    if item: db.session.delete(item); db.session.commit()
    return jsonify({"status": "removed"})

# ═════════════════════════════════════════════════════════════════════════════
# HISTORY
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/history")
@login_required
def history_page():
    u = current_user(); seen = set(); items = []
    for h in reversed(u.history):
        if h.tmdb_id not in seen:
            seen.add(h.tmdb_id)
            items.append({"id": h.tmdb_id, "title": h.title, "poster": h.poster, "rating": h.rating, "type": h.media_type})
    return render_template("index.html", page_title="Watch History", page_items=items,
        active_page="history", active_genre=None, infinite_type=None, infinite_genre=None,
        searched_movie_details=None, recommended_movies=[], recommended_series=[],
        selected_movie=None, carousels={})

@app.route("/history/add", methods=["POST"])
def history_add():
    uid = session.get("user_id")
    if not uid: return jsonify({"status": "guest"})
    data = request.get_json()
    db.session.add(HistoryItem(user_id=uid, tmdb_id=int(data.get("id",0)),
        media_type=data.get("type","movie"), title=data.get("title",""),
        poster=data.get("poster",""), rating=float(data.get("rating",0) or 0)))
    db.session.commit(); return jsonify({"status": "saved"})

# ═════════════════════════════════════════════════════════════════════════════
# MOVIE / TV MODAL  (AJAX — cached)
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/movie_info", methods=["POST"])
def movie_info():
    data    = request.get_json()
    item_id = data.get("movie_id")
    m_type  = data.get("media_type", "movie")

    cache_key = f"movie_info:{item_id}"
    cached = cache_get(cache_key)
    if cached: return jsonify(cached)

    # Confirm type from catalogue
    if master_df is not None and item_id:
        try:
            match = master_df[master_df["id"] == int(item_id)]
            if not match.empty: m_type = match.iloc[0]["type"]
        except Exception: pass

    res = safe_get(f"{TMDB_BASE}/{m_type}/{item_id}", params={
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,videos,watch/providers,external_ids"
    })
    if not res: return jsonify({"error": "Not found"}), 404

    j  = res.json()
    pp = j.get("poster_path")

    # Seasons (separate lightweight call for TV)
    seasons = []
    if m_type == "tv":
        for s in j.get("seasons", []):
            if s.get("season_number", 0) == 0: continue
            seasons.append({
                "season_number": s["season_number"],
                "name":          s.get("name", f"Season {s['season_number']}"),
                "episode_count": s.get("episode_count", 0),
                "air_date":      s.get("air_date", ""),
                "poster":        f"https://image.tmdb.org/t/p/w185{s['poster_path']}" if s.get("poster_path") else None,
            })

    # Watch providers
    watch_providers_raw = j.get("watch/providers", {}).get("results", {}).get("IN", {})
    def fmt_prov(lst):
        return [{"provider_name": p["provider_name"],
                 "logo_path": f"https://image.tmdb.org/t/p/w92{p['logo_path']}"}
                for p in (lst or [])]

    # Build deep links for Indian platforms
    title_q = requests.utils.quote(j.get("title") or j.get("name",""))
    watch_links = []
    for p in watch_providers_raw.get("flatrate", []):
        name = p["provider_name"].lower()
        link = None
        if "netflix"  in name: link = f"https://www.netflix.com/search?q={title_q}"
        elif "prime"  in name or "amazon" in name: link = f"https://www.primevideo.com/search/ref=atv_nb_sr?phrase={title_q}"
        elif "disney" in name or "hotstar" in name: link = f"https://www.hotstar.com/in/search?q={title_q}"
        elif "apple"  in name: link = f"https://tv.apple.com/search?term={title_q}"
        elif "zee"    in name: link = f"https://www.zee5.com/search?q={title_q}"
        elif "jio"    in name: link = f"https://www.jiocinema.com/search/{title_q}"
        elif "sony"   in name: link = f"https://www.sonyliv.com/search?query={title_q}"
        if link:
            watch_links.append({"name": p["provider_name"], "url": link,
                                 "logo": f"https://image.tmdb.org/t/p/w92{p['logo_path']}"})

    details = {
        "title":        j.get("title") or j.get("name"),
        "overview":     j.get("overview"),
        "poster":       f"https://image.tmdb.org/t/p/w500{pp}" if pp else PLACEHOLDER_POSTER,
        "rating":       j.get("vote_average"),
        "release_date": j.get("release_date") or j.get("first_air_date"),
        "runtime":      j.get("runtime") or (j.get("episode_run_time") or [None])[0],
        "genres":       [g["name"] for g in j.get("genres", [])],
        "cast": [{"id": c.get("id"), "name": c["name"], "character": c.get("character",""),
                  "profile": f"https://image.tmdb.org/t/p/w185{c['profile_path']}" if c.get("profile_path") else None}
                 for c in j.get("credits", {}).get("cast", [])[:12]],
        "videos":       j.get("videos", {}).get("results", []),
        "providers":    {"flatrate": fmt_prov(watch_providers_raw.get("flatrate",[])),
                         "rent":     fmt_prov(watch_providers_raw.get("rent",[])),
                         "buy":      fmt_prov(watch_providers_raw.get("buy",[]))},
        "watch_links":  watch_links,
        "type":         m_type,
        "seasons":      seasons,
        "num_seasons":  j.get("number_of_seasons"),
        "num_episodes": j.get("number_of_episodes"),
    }
    cache_set(cache_key, details, 3600)
    return jsonify(details)

@app.route("/api/season/<int:tv_id>/<int:season_num>")
def api_season(tv_id, season_num):
    cache_key = f"season:{tv_id}:{season_num}"
    cached = cache_get(cache_key)
    if cached: return jsonify(cached)
    r = safe_get(f"{TMDB_BASE}/tv/{tv_id}/season/{season_num}", params={"api_key": TMDB_API_KEY})
    if not r: return jsonify({"episodes": []})
    data = r.json()
    episodes = [{"episode_number": ep.get("episode_number"), "name": ep.get("name"),
                 "overview": ep.get("overview"), "air_date": ep.get("air_date"),
                 "runtime": ep.get("runtime"),
                 "still": f"https://image.tmdb.org/t/p/w300{ep['still_path']}" if ep.get("still_path") else None}
                for ep in data.get("episodes", [])]
    result = {"episodes": episodes, "season_name": data.get("name", f"Season {season_num}")}
    cache_set(cache_key, result, 3600)
    return jsonify(result)

# ═════════════════════════════════════════════════════════════════════════════
# ACTOR FILMOGRAPHY
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/api/actor/<int:actor_id>")
def api_actor(actor_id):
    cache_key = f"actor:{actor_id}"
    cached = cache_get(cache_key)
    if cached: return jsonify(cached)

    r = safe_get(f"{TMDB_BASE}/person/{actor_id}", params={
        "api_key": TMDB_API_KEY,
        "append_to_response": "combined_credits"
    })
    if not r: return jsonify({"movies": [], "tv": []})

    j = r.json()
    known_for = j.get("known_for_department", "Acting")

    cast_credits = j.get("combined_credits", {}).get("cast", [])

    def fmt(item, mtype):
        pp  = item.get("poster_path")
        yr  = (item.get("release_date") or item.get("first_air_date") or "")[:4]
        return {
            "id":     item["id"],
            "title":  item.get("title") or item.get("name", "Unknown"),
            "poster": f"https://image.tmdb.org/t/p/w300{pp}" if pp else "",
            "rating": round(item.get("vote_average", 0), 1),
            "year":   yr,
        }

    # Filter, deduplicate, sort by popularity
    seen = set()
    movies, tv = [], []
    for item in sorted(cast_credits, key=lambda x: x.get("popularity", 0), reverse=True):
        iid = item.get("id")
        mt  = item.get("media_type")
        if not iid or iid in seen or mt not in ("movie", "tv"): continue
        # Skip very low vote count / unreleased
        if item.get("vote_count", 0) < 10: continue
        seen.add(iid)
        if mt == "movie" and len(movies) < 40: movies.append(fmt(item, "movie"))
        elif mt == "tv"  and len(tv)     < 40: tv.append(fmt(item, "tv"))
        if len(movies) >= 40 and len(tv) >= 40: break

    result = {"movies": movies, "tv": tv, "known_for": known_for}
    cache_set(cache_key, result, 3600)
    return jsonify(result)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)