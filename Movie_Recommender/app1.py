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
            0.50 * plot_score +
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
    return df.iloc[top_ids][['title', 'year', 'genre', 'imdb_rating', 'metascore','img_url','plot']], None

# Streamlit UI
st.title("🎬 Movie Recommender")
#df
listTabs = ["Tab 1", "Tab 2", "Tab 3"]
# Add em-spaces for wider spacing
tabs = st.tabs([s.center(15, "\u2001") for s in listTabs]) 

with tabs[0]:
    movie_list = sorted(df['title'].unique())
    selected_movie = st.selectbox("Select a movie:", movie_list)

    top_k = st.slider("Number of recommendations", 1, 20, 5)

    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            results, error = recommend_hybrid(selected_movie, df, top_k=top_k)
        if error:
            st.error(error)
        else:
            #st.subheader(f"Top {top_k} movies similar to '{selected_movie}':")
            #st.dataframe(results.reset_index(drop=True))
            for i in range(top_k):
                with st.container():
                # Create columns: image (1/4 width), text (3/4 width)
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.image(results.iloc[i]["img_url"], width=100)
                    with col2:
                        #st.markdown(f"<p style='font-size:24px; font-weight:bold;'>{results.iloc[i]['title']}\
                         #           <span style='padding-left: 32px;font-size: 20px'>{results.iloc[i]['year']}</span></p>",
                          #            unsafe_allow_html=True)
                        #st.write(results.iloc[i]['imdb_rating']+"          "+str(results.iloc[i]['metascore'])+"          "+str(results.iloc[i]['year']))
                        #the below code is used to display the title and year in a single line with the year at the end always
                        #col_title, col_year = st.columns([5, 1])
                        #with col_title:
                        #    st.markdown(f"<p style='font-size: 22px; font-weight: bold;'>{results.iloc[i]['title']}</p>", unsafe_allow_html=True)
                        #with col_year:
                        #    st.markdown(f"<p style='font-size: 16px; color: gray; text-align: right;'>{results.iloc[i]['year']}</p>", unsafe_allow_html=True)
                        #the below code is used to display the title and year in a single line with proper alignment
                        #st.markdown(f"""
                         #       <div style='display: flex; align-items: baseline; gap: 10px;'>
                          #          <span style='font-size: 22px; font-weight: bold;'>{results.iloc[i]['title']}</span>
                           #         <span style='font-size: 16px; color: gray;'>({results.iloc[i]['year']})</span>
                            #    </div>
                            #""", unsafe_allow_html=True)
                        st.markdown(
                                f"""
                                <div style="
                                    display: flex;
                                    align-items: baseline;
                                    justify-content: space-between;
                                    gap: 10px;
                                ">
                                    <div title="{results.iloc[i]['title']}"
                                    style="
                                        font-size: 22px;
                                        font-weight: bold;
                                        white-space: nowrap;
                                        overflow: hidden;
                                        text-overflow: ellipsis;
                                        flex-grow: 1;
                                    ">
                                        {results.iloc[i]['title']}
                                    </div>
                                    <div style="
                                        font-size: 16px;
                                        color: gray;
                                        white-space: nowrap;
                                        flex-shrink: 0;
                                    ">
                                        {results.iloc[i]['year']}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        coli1, coli2, coli3 = st.columns([2, 1, 1])
                        coli1.write(f"**{results.iloc[i]['genre']}**")
                        coli2.write(f"IMDb Rating: {results.iloc[i]['imdb_rating']}")
                        coli3.write(f"Metascore: {results.iloc[i]['metascore']}")
                        st.markdown(
                            f"""
                            <p title='{results.iloc[i]['plot']}'
                              style='
                                display: -webkit-box;
                                -webkit-line-clamp: 2;
                                -webkit-box-orient: vertical;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                font-size: 15px;
                                line-height: 1.5em;
                                max-height: 3em;
                            '>
                                {results.iloc[i]['plot']}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
                st.markdown('<div></div>',unsafe_allow_html=True)
with tabs[1]:
    st.write("This is the Dog tab content.")
with tabs[2]:
    st.write("This is the Owl tab content.")
