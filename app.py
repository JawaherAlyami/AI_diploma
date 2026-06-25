import os
import kagglehub
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Steam Games EDA",
    layout="wide"
)

st.title("🎮 Steam Games Exploratory Data Analysis")

# Load Dataset
@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "fronkongames/steam-games-dataset"
    )

    csv_path = os.path.join(path, "games.csv")

    return pd.read_csv(csv_path, index_col=False)

df = load_data()

# Data Cleaning
cols_to_drop = [
    'About the game', 'Header image', 'Website', 'Support url',
    'AppID', 'Support email', 'Metacritic url', 'Notes',
    'Screenshots', 'Achievements', 'Movies', 'Score rank',
    'Reviews', 'Developers', 'Publishers',
    'Supported languages', 'Full audio languages',
    'Windows', 'Mac', 'Linux',
    'Required age', 'DiscountDLC count',
    'Average playtime two weeks',
    'Average playtime forever',
    'Median playtime two weeks',
    'Median playtime forever',
    'Categories',
    'User score'
]

df = df.drop(columns=cols_to_drop)

df = df.drop_duplicates()

df = df.dropna(subset=['Genres'])

df['Tags'] = df['Tags'].fillna(df['Genres'])

# Encode to 1 and 0
df['Metacritic score'] = df['Metacritic score'].astype(int)

# Bar Chart
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select Section",
    [
        "Dataset Overview",
        "Genre Distribution",
        "Price vs Tags",
        "Genre Engagement",
        "Rating Analysis",
        "Quality vs Popularity",
        "Insights"
    ]
)

# Dataset Overview
if section == "Dataset Overview":

    st.header("The EDA workflow:")
    st.markdown("""
1. Load and look at the data
2. Understand the columns
3. Clean the data (missing values)
4. Explore one column at a time (distributions)
5. Explore relationships between columns
6. Write down all insights""")
    st.header("Dataset Overview")
    st.markdown("""
    More metadata about the dataset:

    Name: Game name

    Release date: When this game relased

    Estimated owners: The estimated number of Steam accounts that own a game.

    Peak CCU: The highest number of players played at the same time game.

    Price: The game price.

    Metacritic score: In gaming is a weighted average of reviews from major professional gaming critics.

    Positive: The positive reviews about each game at row level.

    Negative: The negative reviews about each game at row level.

    Recommendations: The count of recommendations at row level.

    Genres: The geres of modes or player counts. How many people can play at once and interact type.

    Tags: This column contines features of the game
""")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(f"Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("First Rows")
    st.dataframe(df.head(20))

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={
                "index":"Column",
                0:"Null Count"
            }
        )
    )

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())


# Genre Distribution

elif section == "Genre Distribution":

    st.header("Top 10 Categories")

    genre_df = df.copy()

    genre_df['Genres'] = (
        genre_df['Genres']
        .astype(str)
        .str.replace(r'[\[\]]', '', regex=True)
        .str.split(',')
    )

    genre_df = genre_df.explode('Genres')

    genre_df['Genres'] = genre_df['Genres'].str.strip()

    genre_counts = (
        genre_df['Genres']
        .value_counts()
        .reset_index()
    )

    genre_counts.columns = ['Genre', 'Count']

    genre_counts = genre_counts.head(10)

    fig = px.pie(
        genre_counts,
        names='Genre',
        values='Count',
        title='Top 10 Categories'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""Insight
Single-player functionality is the primary driver of both game availability and player engagement on Steam, accounting for over 28% of all titles.""")


# Price vs Tags

elif section == "Price vs Tags":

    st.header("Average Price by Tag")

    tag_price = (
        df.groupby('Tags')['Price']
        .mean()
        .reset_index()
    )

    tag_price.columns = ['Tag', 'Average Price']

    tag_price = tag_price.sort_values(
        'Average Price',
        ascending=False
    ).head(15)

    fig = px.bar(
        tag_price,
        x='Average Price',
        y='Tag',
        labels={"Average_Price": "Average Price ($)", "Tag": "Tag / Feature"},
        orientation='h',
        color='Average Price'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""Insight
Niche utility software, creative development tools, and highly bundled simulation packages command the highest premium pricing on the platform, with average price points scaling up to \$200. And the lower average is 50\$""")

# Genre Engagement
elif section == "Genre Engagement":

    st.header("Genre Engagement Analysis")

    engagement_df = df.copy()

    engagement_df['Genres'] = (
        engagement_df['Genres']
        .astype(str)
        .str.replace(r'[\[\]]', '', regex=True)
        .str.split(',')
    )

    engagement_df = engagement_df.explode('Genres')

    engagement_df['Genres'] = (
        engagement_df['Genres']
        .str.strip()
    )

    engagement_df['Total_Reviews'] = (
        engagement_df['Positive']
        + engagement_df['Negative']
    )

    genre_engagement = (
        engagement_df.groupby('Genres')
        .agg(
            Total_Reviews=('Total_Reviews','sum'),
            Recommendations=('Recommendations','sum'),
            Games=('Name','count')
        )
        .reset_index()
        .sort_values(
            'Total_Reviews',
            ascending=False
        )
    )

    st.dataframe(genre_engagement.head(15))

    fig = px.bar(
        genre_engagement.head(10),
        x='Genres',
        y='Total_Reviews',
        title='Top Genres by Reviews'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""Insight
                Meta-features like Steam Achievements, Family Sharing, and Steam Trading Cards represent the most dominant ecosystem features, appearing heavily among the platform's most highly reviewed and visible games.""")
# Rating Analysis
elif section == "Rating Analysis":

    st.header("Approval Rate Analysis")

    rating_df = df.copy()

    rating_df['Total_Reviews'] = (
        rating_df['Positive']
        + rating_df['Negative']
    )

    rating_df['Approval_Rate'] = (
        rating_df['Positive']
        / (rating_df['Total_Reviews'] + 1)
    ) * 100

    top_rated = (
        rating_df
        .sort_values(
            'Approval_Rate',
            ascending=False
        )
        .head(10)
    )

    low_rated = (
        rating_df
        .sort_values(
            'Approval_Rate',
            ascending=True
        )
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:
     st.subheader("Top Rated Games")

     fig = px.bar(
        top_rated,
        x="Name",
        y="Approval_Rate",
        title="Top 10 Games by Approval Rate",
        labels={
            "Name": "Game",
            "Approval_Rate": "Approval Rate (%)"
        },
        color="Approval_Rate"
     )

     fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
     )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""Insight
The highest consumer approval rates on Steam are heavily dominated by niche, targeted indie genres (such as visual novels and adult-themed titles), where tightly focused developer-to-audience alignment yields concentrated positive feedback.""")


elif section == "Quality vs Popularity":
    st.subheader("Quality vs Popularity")

    # 1. Clean the data for this specific analysis
    df_quality = df.copy()
    df_quality = df_quality.dropna(subset=['Metacritic score', 'Peak CCU'])
    df_quality = df_quality[df_quality['Peak CCU'] > 0]
    
    df_quality['Review Quality'] = df_quality['Metacritic score'].map({0: 'Low/Avg Quality', 1: 'High Quality'})

    metacritic_ccu = (
        df_quality.groupby('Review Quality')['Peak CCU']
        .mean()
        .reset_index()
    )
    
    fig = px.bar(
        metacritic_ccu,
        x='Review Quality',
        y='Peak CCU',
        title='Average Peak CCU by Review Quality',
        labels={
            'Review Quality': 'Review Quality Category',
            'Peak CCU': 'Average Peak CCU'
        },
        color='Review Quality',
        color_discrete_map={'Low/Avg Quality': '#e74c3c', 'High Quality': '#2ecc71'}
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""Insight
     The quality helps determine whether critically acclaimed games tend to attract larger player bases on Steam.""")
# Insights
elif section == "Insights":

    st.header("Key Insights")

    st.markdown("""
    ### Insight 1
    Single-player functionality is the primary driver of both game availability and player engagement on Steam, accounting for over 28% of all titles (Usually Developers develop games within these catagories).

    ### Insight 2
    Niche utility software, creative development tools, and highly bundled simulation packages command the highest premium pricing on the platform, with average price points scaling up to \$200. And the lower average is 50\$.(These factors affected the price).

    ### Insight 3
    Meta-features like Steam Achievements, Family Sharing, and Steam Trading Cards represent the most dominant ecosystem features, appearing heavily among the platform's most highly reviewed and visible games..

    ### Insight 4
    Meta-features like Steam Achievements, Family Sharing, and Steam Trading Cards represent the most dominant ecosystem features, appearing heavily among the platform's most highly reviewed and visible games.
                
     ### Insight 5
     The quality helps determine whether critically acclaimed games tend to attract larger player bases on Steam.
    """)

    # 