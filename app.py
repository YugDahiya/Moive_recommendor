from flask import Flask, render_template, request
import pickle
import requests
import pandas as pd

app = Flask(__name__)


def fetch_poster(movie_id):
    """Fetches the movie poster URL from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url, timeout=5)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster: {e}")
    return "https://placehold.co/500x750/333/FFFFFF?text=No+Poster"


try:
    movies_dict = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    # Ensure necessary columns are numeric
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce')
    
    similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
except FileNotFoundError:
    print("Model files not found. Please run the data processing notebook first.")
    movies, similarity = None, None
except Exception as e:
    print(f"Error loading pickle files: {e}")
    movies, similarity = None, None


def get_movie_details(movie_series):
    """Helper function to format movie data."""
    return {
        'title': movie_series.title,
        'poster': fetch_poster(movie_series.movie_id),
        'year': int(movie_series.year) if pd.notna(movie_series.year) else "N/A",
        'rating': f"{movie_series.vote_average:.1f}"
    }


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
        recommendations.append(get_movie_details(movie_data))
    return recommendations


def get_curated_movies(sort_by, n=20): # Increased to 20 for carousel
    """Gets a list of movies sorted by a specific column."""
    
    if sort_by == 'vote_average':
        df_filtered = movies[(movies['vote_average'] <= 9.5) & (movies['vote_average'] > 0)]
    else:
        df_filtered = movies
        
    df_sorted = df_filtered.sort_values(sort_by, ascending=False).dropna(subset=[sort_by])
    
    curated_list = []
    for _, row in df_sorted.head(n).iterrows():
        curated_list.append(get_movie_details(row))
    return curated_list

def get_movies_by_genre(genre_name, n=20): # Increased to 20 for carousel
    """Gets a list of movies for a specific genre."""
    if 'genres' not in movies.columns:
        print(f"Warning: 'genres' column not found in dataframe.")
        return []
        
    try:
        genre_movies = movies[movies['genres'].apply(lambda x: genre_name in x if isinstance(x, list) else False)]
    except TypeError:
        try:
            genre_movies = movies[movies['genres'].str.contains(genre_name, na=False)]
        except:
            print(f"Error processing 'genres' column for genre: {genre_name}")
            return []
    
    df_sorted = genre_movies.sort_values('vote_average', ascending=False)
    
    curated_list = []
    for _, row in df_sorted.head(n).iterrows():
        curated_list.append(get_movie_details(row))
    return curated_list


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if movies is None:
        return "<h3>Model files not found. Please make sure 'artifacts' folder is present.</h3>"

    movie_list = movies['title'].values
    recommendations = []
    selected_movie = None
    searched_movie_details = None # NEW: For searched movie
    top_rated = []
    action_movies = []
    comedy_movies = []
    drama_movies = []
    scifi_movies = []

    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        recommendations = recommend(selected_movie)
        
        # NEW: Get details for the movie that was searched
        if selected_movie:
            try:
                movie_data = movies[movies['title'] == selected_movie].iloc[0]
                searched_movie_details = get_movie_details(movie_data)
            except IndexError:
                print(f"Searched movie '{selected_movie}' not found in dataframe.")
        
    else:
        # GET request: Load landing page content
        top_rated = get_curated_movies('vote_average', 20)
        action_movies = get_movies_by_genre('Action', 20)
        comedy_movies = get_movies_by_genre('Comedy', 20)
        drama_movies = get_movies_by_genre('Drama', 20)
        scifi_movies = get_movies_by_genre('Science Fiction', 20)

    return render_template('index.html',
                           movie_list=movie_list,
                           recommendations=recommendations,
                           selected_movie=selected_movie,
                           searched_movie_details=searched_movie_details, # NEW
                           top_rated=top_rated,
                           action_movies=action_movies,
                           comedy_movies=comedy_movies,
                           drama_movies=drama_movies,
                           scifi_movies=scifi_movies
                           )


if __name__ == '__main__':
    app.run(debug=True)