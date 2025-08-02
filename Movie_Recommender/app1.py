import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Movie_Recommender/final_movie_with_Sent_embedding_cleaned_16K.csv")

    # Parse plot_embedding from string to np.array
    #df["plot_embedding"] = df["plot_embedding"].apply(ast.literal_eval)
    #df["plot_embedding"] = df["plot_embedding"].apply(np.array)
    df["plot_embedding"] = df["plot_embedding"].apply(lambda x : np.fromstring(x.strip('[]'), sep=' '))
    # Filter movies with votes > 5000
    #df = df[df['votes'] > 5000].reset_index(drop=True)

    # Normalize votes for popularity scoring
    scaler = MinMaxScaler()
    df['votes'] = scaler.fit_transform(df[['votes']])

    return df

df = load_data()

# Hybrid scoring function
def hybrid_scores(selected_idx, df):
    selected_movie = df.iloc[selected_idx]
    selected_emb = selected_movie['plot_embedding'].reshape(1, -1)
    all_embeddings = np.stack(df['plot_embedding'].values)

    # Compute cosine similarities for plot
    plot_sims = cosine_similarity(selected_emb, all_embeddings)[0]

    scores = []
    for i, movie in df.iterrows():
        if i == selected_idx:
            continue

        # Genre overlap
        genres_a = set(str(selected_movie['genre']).split(', '))
        genres_b = set(str(movie['genre']).split(', '))
        genre_overlap = len(genres_a & genres_b) / max(len(genres_a), 1)

        # Year
        year_score = max(0, 1 - abs(selected_movie['year'] - movie['year']) / 10)

        # Certificate
        certificate_score = 1 if selected_movie.get('certificates') == movie.get('certificates') else 0

        # Metascore
        if pd.isna(selected_movie['metascore']) or pd.isna(movie['metascore']):
            metascore_score = 0.5
        else:
            metascore_score = 1 - abs(selected_movie['metascore'] - movie['metascore']) / 100

        # IMDb rating
        if pd.isna(selected_movie['imdb_rating']) or pd.isna(movie['imdb_rating']):
            imdb_score = 0.5
        else:
            imdb_score = 1 - abs(selected_movie['imdb_rating'] - movie['imdb_rating']) / 10

        # Popularity
        pop_score = min(np.log1p(movie['votes']) / np.log1p(selected_movie['votes']), 1.0) if selected_movie['votes'] > 0 else 0

        # Plot similarity
        plot_score = plot_sims[i]

        # Final hybrid score (tune weights if desired)
        final_score = (
            0.30 * plot_score +
            0.15 * genre_overlap +
            0.05 * year_score +
            0.10 * certificate_score +
            0.10 * metascore_score +
            0.20 * imdb_score +
            0.10 * pop_score
        )

        scores.append((i, final_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# Recommend movies
def recommend_hybrid(title, df, top_k=5):
    title_to_index = {t.lower(): i for i, t in enumerate(df['title'])}
    idx = title_to_index.get(title.lower())
    if idx is None:
        return None, f"Movie '{title}' not found."

    ranked = hybrid_scores(idx, df)
    top_ids = [i for i, _ in ranked[:top_k]]
    return df.iloc[top_ids][['title', 'year', 'genre', 'imdb_rating', 'metascore']], None

# Streamlit UI
st.title("🎬 Hybrid Movie Recommender")

movie_list = sorted(df['title'].unique())
selected_movie = st.selectbox("Select a movie:", movie_list)

top_k = st.slider("Number of recommendations", 1, 20, 5)

if st.button("Get Recommendations"):
    with st.spinner("Finding similar movies..."):
        results, error = recommend_hybrid(selected_movie, df, top_k=top_k)
    if error:
        st.error(error)
    else:
        st.subheader(f"Top {top_k} movies similar to '{selected_movie}':")
        st.dataframe(results.reset_index(drop=True))

