import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ast
import html

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
    df['genre'] = df['genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df

df = load_data()

# Hybrid scoring function
def hybrid_scores(selected_idx, df, plot_imp, genre_imp, year_imp, cert_imp, metascore_imp, imdb_imp, pop_imp):
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
            plot_imp * plot_score +
            genre_imp * genre_overlap +
            year_imp * year_score +
            cert_imp * certificate_score +
            metascore_imp * metascore_score +
            imdb_imp * imdb_score +
            pop_imp * pop_score
        )

        scores.append((i, final_score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# Recommend movies
def recommend_hybrid(title, df, top_k=5,plot_imp=0.3, genre_imp=0.15, year_imp=0.05, cert_imp=0.10, metascore_imp=0.10, imdb_imp=0.20, pop_imp=0.10):
    title_to_index = {t.lower(): i for i, t in enumerate(df['title'])}
    idx = title_to_index.get(title.lower())
    if idx is None:
        return None, f"Movie '{title}' not found."

    ranked = hybrid_scores(idx, df, plot_imp=plot_imp, genre_imp=genre_imp, year_imp=year_imp, cert_imp=cert_imp, metascore_imp=metascore_imp, imdb_imp=imdb_imp, pop_imp=pop_imp)
    top_ids = [i for i, _ in ranked[:top_k]]
    return df.iloc[top_ids][['title', 'year', 'genre', 'imdb_rating', 'metascore','img_url','plot']], None

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("""
                <style>
                    .stMainBlockContainer  {
                        padding-top: 1.4rem !important;
                    }
                    button[data-baseweb="tab"] .st-emotion-cache-17r1dd6.e1fxfrsf0 p {
                        font-size: 16px;
                    }
                </style>""", unsafe_allow_html=True)
st.title("🎬 Movie Recommender")
#df
listTabs = ["Default Recommender", "Custom Recommender", "🔍 Discover Movies"]
# Add em-spaces for wider spacing
tabs = st.tabs([s.center(37, "\u2001") for s in listTabs]) 

st.markdown("""
            <style>
                .stForm {
                border: none;
                padding: 0px !important;
            }
            .st-cb{
                padding-top: 0px !important;
            }
            </style>""", unsafe_allow_html=True)

with tabs[0]:
    movie_list = sorted(df['title'].unique())
    st.markdown("""
                <div style="margin-top:1rem"></div>""", unsafe_allow_html=True)
    selected_movie = st.selectbox("Select a movie:", movie_list, key="movie_select")

    top_k = st.slider("Number of recommendations", 1, 20, 5,key="slider_select")
    #current_tab = tabs[0]

        # Only inject style when the tab is switched
    
    with st.form("recommendation_form_tab"):
        
                # 🚀 This will ONLY trigger a rerun on click
        submitted = st.form_submit_button("Get Recommendations")
    if submitted:
        with st.spinner("Finding similar movies..."):
            results, error = recommend_hybrid(selected_movie, df, top_k=top_k)
        if error:
            st.error(error)
        else:
            #st.subheader(f"Top {top_k} movies similar to '{selected_movie}':")
            #st.dataframe(results.reset_index(drop=True))
            #for i in range(top_k):

            num_columns = 2

            results.loc[pd.isna(results['metascore']), 'metascore'] = "Not Available"

            for i in range(0, len(results), num_columns):
                cols = st.columns(num_columns)
                
                for j in range(num_columns):
                    if i + j < len(results):
                        with cols[j]:
                            with st.container():
                                if j==0:
                                    col1, col2,col3 = st.columns([0.5, 3,0.1])
                                else:
                                    col3,col1, col2 = st.columns([0.1,0.5, 3])
                                with col1:
                                    st.image(results.iloc[i+j]["img_url"], width=100)
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
                                    title_text = results.iloc[i+j]['title']
                                    title_text=html.escape(title_text) 
                                    st.markdown(
                                            f"""
                                            <div style="
                                                display: flex;
                                                align-items: baseline;
                                                justify-content: space-between;
                                                gap: 10px;
                                            ">
                                                <div title="{title_text}"
                                                style="
                                                    font-size: 22px;
                                                    font-weight: bold;
                                                    white-space: nowrap;
                                                    overflow: hidden;
                                                    text-overflow: ellipsis;
                                                    flex-grow: 1;
                                                ">
                                                    {title_text}
                                                </div>
                                                <div style="
                                                    font-size: 16px;
                                                    color: gray;
                                                    white-space: nowrap;
                                                    flex-shrink: 0;
                                                ">
                                                    {results.iloc[i+j]['year']}
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    coli1, coli2, coli3 = st.columns([2.3, 1.2, 1.5])
                                    with coli1:
                                        st.markdown(f"<p style='font-size:14px;font-weight:400'>{results.iloc[i+j]['genre']}</p>", unsafe_allow_html=True)
                                    with coli2:
                                        st.markdown(f"<p style='font-size:14px;font-weight:400'>IMDb Rating: {results.iloc[i+j]['imdb_rating']} ⭐</p>", unsafe_allow_html=True)
                                    with coli3:
                                        st.markdown(f"<p style='font-size:14px;font-weight:400'>Metascore: {results.iloc[i+j]['metascore']}</p>", unsafe_allow_html=True)
                                    plot_text = results.iloc[i+j]['plot']
                                    escaped_plot = html.escape(plot_text) 
                                    st.markdown(
                                        f"""
                                        <p title='{escaped_plot}'
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
                                            {escaped_plot}
                                        </p>
                                        """,
                                        unsafe_allow_html=True
                                    )
                        st.markdown('<div></div>',unsafe_allow_html=True)
with tabs[1]:
    movie_list = sorted(df['title'].unique())
    st.markdown("""
                <style>
                    .stForm {
                    border: none;
                    padding: 0px !important;
                }
                .st-cb{
                    padding-top: 0px !important;
                }
                </style>""", unsafe_allow_html=True)
    with st.form("recommendation_form_tab1"):
        selected_movie = st.selectbox("Select a movie:", movie_list, key="movie_select1")
        top_k = st.slider("Number of recommendations", 1, 20, 5, key="slider_select1")

        val1, val2, val3,val4, val5, val6 = st.columns(6)
        with val1:
            plot_imp = st.number_input("Plot Importance", 0.0, 1.0, 0.3, step=0.01, key="plot_imp")
        with val2:
            genre_imp = st.number_input("Genre Importance", 0.0, 1.0, 0.15, step=0.01, key="genre_imp")
        with val3:
            year_imp = st.number_input("Year Importance", 0.0, 1.0, 0.05, step=0.01, key="year_imp")

        #val4, val5, val6 = st.columns(3)
        with val4:
            pop_imp = st.number_input("Popularity Importance", 0.0, 1.0, 0.10, step=0.01, key="pop_imp")
        with val5:
            metascore_imp = st.number_input("Metascore Importance", 0.0, 1.0, 0.10, step=0.01, key="metascore_imp")
        with val6:
            imdb_imp = st.number_input("IMDb Rating Importance", 0.0, 1.0, 0.20, step=0.01, key="imdb_imp")

        # 🚀 This will ONLY trigger a rerun on click
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        #if st.button("Get Recommendations" ,key="recommend_button1"):
        with st.spinner("Finding similar movies..."):
            results, error = recommend_hybrid(selected_movie, df, top_k=top_k ,plot_imp=plot_imp, genre_imp=genre_imp, year_imp=year_imp, cert_imp=0.10, metascore_imp=metascore_imp, imdb_imp=imdb_imp, pop_imp=pop_imp)
        
        #if error:
        #   st.error(error)
        #else:
            #st.subheader(f"Top {top_k} movies similar to '{selected_movie}':")
            #st.dataframe(results.reset_index(drop=True))
            if not error:
                st.session_state['recommendations_tab2'] = results
                st.session_state['recommendations_error_tab2'] = None
            else:
                st.session_state['recommendations_error_tab2'] = error
                st.session_state['recommendations_tab2'] = None

    # Show previous results if they exist
    if 'recommendations_error_tab2' in st.session_state and st.session_state['recommendations_error_tab2']:
        st.error(st.session_state['recommendations_error_tab2'])
    elif 'recommendations_tab2' in st.session_state and st.session_state['recommendations_tab2'] is not None:
        results = st.session_state['recommendations_tab2']
        num_columns = 2
        results.loc[pd.isna(results['metascore']), 'metascore'] = "Not Available"

        for i in range(0, len(results), num_columns):
            cols = st.columns(num_columns)
            
            for j in range(num_columns):
                if i + j < len(results):
                    with cols[j]:
                        with st.container():
                            if j==0:
                                col1, col2,col3 = st.columns([0.5, 3,0.1])
                            else:
                                col3,col1, col2 = st.columns([0.1,0.5, 3])
                            with col1:
                                st.image(results.iloc[i+j]["img_url"], width=100)
                            with col2:
                                title_text = results.iloc[i+j]['title']
                                title_text=html.escape(title_text) 
                                st.markdown(
                                        f"""
                                        <div style="
                                            display: flex;
                                            align-items: baseline;
                                            justify-content: space-between;
                                            gap: 10px;
                                        ">
                                            <div title="{title_text}"
                                            style="
                                                font-size: 22px;
                                                font-weight: bold;
                                                white-space: nowrap;
                                                overflow: hidden;
                                                text-overflow: ellipsis;
                                                flex-grow: 1;
                                            ">
                                                {title_text}
                                            </div>
                                            <div style="
                                                font-size: 16px;
                                                color: gray;
                                                white-space: nowrap;
                                                flex-shrink: 0;
                                            ">
                                                {results.iloc[i+j]['year']}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                coli1, coli2, coli3 = st.columns([2.3, 1.2, 1.5])
                                with coli1:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>{results.iloc[i+j]['genre']}</p>", unsafe_allow_html=True)
                                with coli2:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>IMDb Rating: {results.iloc[i+j]['imdb_rating']} ⭐</p>", unsafe_allow_html=True)
                                with coli3:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>Metascore: {results.iloc[i+j]['metascore']}</p>", unsafe_allow_html=True)
                                plot_text = results.iloc[i+j]['plot']
                                escaped_plot = html.escape(plot_text) 
                                st.markdown(
                                    f"""
                                    <p title='{escaped_plot}'
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
                                        {escaped_plot}
                                    </p>
                                    """,
                                    unsafe_allow_html=True
                                )
                    st.markdown('<div></div>',unsafe_allow_html=True)
with tabs[2]:
    #st.markdown("### 🔍 Discover Movies")
    
    mode = st.radio("Choose a discovery mode:", ["Random", "Top by Genre", "Top by Year"],horizontal=True, key="discovery_mode")
    #if mode =="Random":
        #sel2,sl3=st.columns([2,12])
        
    #else:
        #sel1,sel2,sel3=st.columns([5,2,7])
        
    num_movies = st.slider("Number of movies to show", 1, 20, 4)
    
    #if mode != "Random":
        #with sel1:
    if mode == "Top by Genre":
        selected_genre = st.multiselect("Select Genre:", sorted(set(sorted(df['genre'].explode().unique()))),max_selections=3, key="genre_select",width=550)
    elif mode == "Top by Year":
        selected_year = st.slider("Select Year:", int(df['year'].min()), int(df['year'].max()), 2020,width=550)
    if mode == "Random":
        st.markdown("""
                <style>
                    .stButton.st-emotion-cache-8atqhb.e1mlolmg0 {
                        margin-top: -1rem;
                        margin-left: 0rem;
                    }
                </style>""", unsafe_allow_html=True)
        sel2, sl3 = st.columns([2, 12])
    elif mode == "Top by Year":
        st.markdown("""
                <style>
                    .stButton.st-emotion-cache-8atqhb.e1mlolmg0 {
                        margin-top: -5.45rem;
                        margin-left: -26rem;
                    }
                </style>""", unsafe_allow_html=True)
        sel3, sel2  = st.columns([5, 2])
    else:
        st.markdown("""
                <style>
                    .stButton.st-emotion-cache-8atqhb.e1mlolmg0 {
                        margin-top: -4.5rem;
                        margin-left: -26rem;
                    }
                </style>""", unsafe_allow_html=True)
        sel3, sel2 = st.columns([5, 2])
    with sel2:
        if st.button("Show Movies", key="discover_button"):
            if mode == "Random":
                results = df.sample(num_movies)
            elif mode == "Top by Genre":
                #filtered_df = df[df['genre'].apply(lambda g_list: selected_genre in g_list)]
                if selected_genre:
                    filtered_df = df[df['genre'].apply(lambda g_list: all(genre in g_list for genre in selected_genre))]
                else:
                    filtered_df = df.copy()
                results = (
                    #df[df['genre'].apply(lambda g: selected_genre in g)]
                    filtered_df
                    .sort_values(by="imdb_rating", ascending=False)
                    .head(num_movies)
                )
            elif mode == "Top by Year":
                results = (
                    df[df['year'] == selected_year]
                    .sort_values(by="imdb_rating", ascending=False)
                    .head(num_movies)
                )
            st.session_state['discovery_results'] = results

    # Display results if available
    if 'discovery_results' in st.session_state:
        results = st.session_state['discovery_results']

        num_columns = 2   
        results.loc[pd.isna(results['metascore']), 'metascore'] = "Not Available"

        for i in range(0, len(results), num_columns):
            cols = st.columns(num_columns)
            
            for j in range(num_columns):
                if i + j < len(results):
                    with cols[j]:
                        with st.container():
                            if j==0:
                                col1, col2,col3 = st.columns([0.5, 3,0.1])
                            else:
                                col3,col1, col2 = st.columns([0.1,0.5, 3])
                            with col1:
                                st.image(results.iloc[i+j]["img_url"], width=100)
                            with col2:
                                title_text = results.iloc[i+j]['title']
                                title_text=html.escape(title_text) 
                                st.markdown(
                                        f"""
                                        <div style="
                                            display: flex;
                                            align-items: baseline;
                                            justify-content: space-between;
                                            gap: 10px;
                                        ">
                                            <div title="{title_text}"
                                            style="
                                                font-size: 22px;
                                                font-weight: bold;
                                                white-space: nowrap;
                                                overflow: hidden;
                                                text-overflow: ellipsis;
                                                flex-grow: 1;
                                            ">
                                                {title_text}
                                            </div>
                                            <div style="
                                                font-size: 16px;
                                                color: gray;
                                                white-space: nowrap;
                                                flex-shrink: 0;
                                            ">
                                                {results.iloc[i+j]['year']}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                coli1, coli2, coli3 = st.columns([2.3, 1.2, 1.5])
                                with coli1:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>{results.iloc[i+j]['genre']}</p>", unsafe_allow_html=True)
                                with coli2:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>IMDb Rating: {results.iloc[i+j]['imdb_rating']} ⭐</p>", unsafe_allow_html=True)
                                with coli3:
                                    st.markdown(f"<p style='font-size:14px;font-weight:400'>Metascore: {results.iloc[i+j]['metascore']}</p>", unsafe_allow_html=True)
                                plot_text = results.iloc[i+j]['plot']
                                escaped_plot = html.escape(plot_text) 
                                st.markdown(
                                    f"""
                                    <p title='{escaped_plot}'
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
                                        {escaped_plot}
                                    </p>
                                    """,
                                    unsafe_allow_html=True
                                )
                    st.markdown('<div></div>',unsafe_allow_html=True)




