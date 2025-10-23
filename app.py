from flask import Flask, render_template, request
import pickle
import requests
import pandas as pd

app = Flask(__name__)

# --- Helper Function to Fetch Poster ---
def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url,timeout=5)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"


# --- Load the Model Files ---
try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
except FileNotFoundError:
    print("Model files not found. Please run the data processing notebook first.")
    movies, similarity = None, None


# --- Recommendation Logic ---
def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommendations = []

    for i in distances[1:6]:
        movie_data = movies.iloc[i[0]]
        recommendations.append({
            'title': movie_data.title,
            'poster': fetch_poster(movie_data.movie_id),
            'year': int(movie_data.year) if pd.notna(movie_data.year) else "N/A",
            'rating': f"{movie_data.vote_average:.1f}"
        })
    return recommendations


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if movies is None:
        return "<h3>Model files not found. Please make sure 'artifacts' folder is present.</h3>"

    movie_list = movies['title'].values
    recommendations = []
    selected_movie = None

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        recommendations = recommend(selected_movie)

    return render_template('index.html', movie_list=movie_list,
                           recommendations=recommendations,
                           selected_movie=selected_movie)


if __name__ == '__main__':
    app.run(debug=True)