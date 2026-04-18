import streamlit as st
import pandas as pd
import numpy as np
from als_model import train_als_model, recommend_als
from spark_processing import get_spark_dataframe

from data_preprocessing import load_data, create_user_item_matrix
from models import compute_item_similarity, recommend_items, train_svd_model, recommend_svd


# PAGE CONFIGURATION 

# This sets the browser tab title, icon, and layout of the app
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎧",
    layout="wide"
)


# UI THEME 

# Custom CSS is used to give a Spotify-like dark theme to the app
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: white;
    }

    h1, h2, h3, h4 {
        color: #1DB954;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    div.stButton > button {
        background-color: #1DB954;
        color: black;
        font-weight: bold;
        border-radius: 20px;
        height: 3em;
        width: 100%;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #17a74a;
        color: black;
    }

    .spotify-card {
        background-color: #181818;
        padding: 18px;
        border-radius: 15px;
        margin-bottom: 12px;
        transition: 0.3s;
    }

    .spotify-card:hover {
        background-color: #282828;
        transform: scale(1.02);
    }

    .metric-box {
        background-color: #181818;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# DATA LOADING

@st.cache_resource
def load_and_prepare_data():
    # This function loads data once and caches it to avoid reloading on every UI interaction
    print("Loading data started")

    # Load ratings dataset
    ratings = load_data("ratings.csv")

    # Take a smaller sample for faster training and UI responsiveness
    ratings = ratings.sample(10000, random_state=42)

    # Load movie metadata and rename column for consistency
    movies = pd.read_csv("movies.csv")
    movies = movies.rename(columns={"movieId": "item_id"})

    # Load Spark dataframe for ALS model
    spark_df = get_spark_dataframe("ratings.csv")

    return ratings, movies, spark_df


# Load data once at app start
data, movies, spark_df = load_and_prepare_data()


# SIDEBAR 

# Sidebar acts as control panel for user interaction
st.sidebar.title("🎛 Control Panel")

# Dropdown to select user
user_ids = sorted(data['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID", user_ids)

# Slider to choose number of recommendations
top_k = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Show dataset statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.write("Total Ratings:", len(data))
st.sidebar.write("Users:", data['user_id'].nunique())
st.sidebar.write("Items:", data['item_id'].nunique())


# HEADER 

# Main title and description
st.title("🎧 Music Recommendation System")
st.markdown("Discover personalized recommendations using advanced ML algorithms.")


# MODEL PREPARATION 

@st.cache_resource
def prepare_cosine_model(data):
    # Prepare user-item matrix and similarity matrix for cosine model
    user_item_matrix = create_user_item_matrix(data)
    similarity_matrix, item_matrix = compute_item_similarity(user_item_matrix)
    return user_item_matrix, similarity_matrix, item_matrix


@st.cache_resource
def prepare_svd_model(data):
    # Train SVD model and return model with RMSE
    model, rmse = train_svd_model(data)
    return model, rmse


# RECOMMENDATION LOGIC 

# Button triggers recommendation generation
if st.button("🎵 Get Recommendations"):

    # Create two columns for layout
    col1, col2 = st.columns([3, 1])

    with col1:

        # Show loading spinner while computing
        with st.spinner("Generating recommendations..."):

            # Train ALS model using Spark dataframe
            als_model = train_als_model(spark_df)

            # Generate recommendations using ALS
            recommendations = recommend_als(
                als_model,
                selected_user,
                spark_df,
                top_k
            )

        # This second call ensures recommendations are available after spinner
        recommendations = recommend_als(
            als_model,
            selected_user,
            spark_df,
            top_k
        )

        # Display recommendations
        st.subheader("🎼 Recommended For You")

        for item in recommendations:
            # Fetch movie title from metadata
            title = movies[movies['item_id'] == item]['title'].values

            if len(title) > 0:
                st.markdown(f"""
                <div class="spotify-card">
                    🎵 <b>{title[0]}</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"🎵 Item ID: {item}")

    with col2:

        # Show model information in side panel
        st.subheader("📈 Model Performance")

        st.markdown("""
        <div class="metric-box">
            <h3>ALS (Spark)</h3>
            <p>Distributed Recommendation Model</p>
        </div>
        """, unsafe_allow_html=True)