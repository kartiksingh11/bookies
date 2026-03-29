import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

class ProfessionalRecommender:
    def __init__(self, listing_path, books_path, description_path):
        # Load and clean data
        self.data = pd.read_csv(listing_path, encoding='latin-1')
        self.books = pd.read_csv(books_path, encoding='latin-1')
        self.desc = pd.read_csv(description_path, encoding='latin-1')
        
        # Pre-process content engine (TF-IDF)
        self.desc['description'] = self.desc['description'].fillna('')
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.desc['description'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        
        # Pre-process correlation engine (Pivot Table)
        self.user_rating_pivot = self.books.pivot_table(
            index='user_id', columns='book_id', values='user_rating'
        )

    def get_recommendations(self, target_book_name, n=5):
        if target_book_name not in self.data['name'].values:
            return "Book not found."

        # 1. Identify Target Metadata
        target_row = self.data[self.data['name'] == target_book_name].iloc[0]
        target_id = target_row['book_id']
        target_author = target_row['author']
        
        # Get content index
        try:
            content_idx = self.desc[self.desc['book_id'] == target_id].index[0]
        except IndexError:
            return "Description data missing for this book."

        # 2. Compute Scores
        sim_scores = self.cosine_sim[content_idx] # Content Score Array
        
        try:
            target_ratings = self.user_rating_pivot[target_id]
            corr_scores = self.user_rating_pivot.corrwith(target_ratings) # Correlation Series
        except KeyError:
            corr_scores = pd.Series(0, index=self.user_rating_pivot.columns)

        # 3. Combine with Metadata Bias
        recommendation_list = []
        for i, row in self.data.iterrows():
            current_id = row['book_id']
            if current_id == target_id:
                continue
            
            # Content Score lookup
            desc_match = self.desc[self.desc['book_id'] == current_id]
            c_score = sim_scores[desc_match.index[0]] if not desc_match.empty else 0
            
            # Correlation Score (normalized to 0-1)
            raw_corr = corr_scores.get(current_id, 0)
            r_score = (np.nan_to_num(raw_corr) + 1) / 2
            
            # Metadata Bias (1.0 if same author)
            m_bias = 1.0 if row['author'] == target_author else 0.0
            
            # Weighted Score: 40% Content, 40% Correlation, 20% Author
            final_score = (c_score * 0.4) + (r_score * 0.4) + (m_bias * 0.2)
            
            recommendation_list.append({
                'name': row['name'],
                'author': row['author'],
                'score': final_score
            })

        # 4. Return top N
        result_df = pd.DataFrame(recommendation_list)
        return result_df.sort_values(by='score', ascending=False).head(n)

# --- STREAMLIT UI ---
st.set_page_config(page_title="BookVault Recommender", page_icon="📚", layout="wide")

# App Styling
st.markdown("""
    <style>
    .stApp {background-color: #f8fafc;}
    h1 {color: #1e3a8a;}
    .rec-card {padding: 15px; border-radius: 10px; border-left: 5px solid #1e3a8a; margin-bottom: 10px; background: white;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def init_engine():
    # Cache the engine to avoid re-calculating matrices on every click
    return ProfessionalRecommender('listing.csv', 'books.csv', 'description.csv')

st.title("🛡️ BookVault: Hybrid AI Recommender")
st.write("Combining NLP Context, Collaborative Intelligence, and Metadata Bias.")

engine = init_engine()

# Sidebar Setup
# --- Updated Sidebar Setup ---
with st.sidebar:
    st.header("Search Engine")
    
    # 1. Drop any NaN values from the name column
    # 2. Convert everything to string just in case
    # 3. Get unique names and sort them
    clean_names = engine.data['name'].dropna().astype(str).unique().tolist()
    unique_books = sorted(clean_names)
    
    selected_book = st.selectbox("Pick a book you enjoyed:", [""] + unique_books)
    num_recommendations = st.slider("How many suggestions?", 1, 10, 5)
# Recommendation Logic
if selected_book:
    with st.spinner("Analyzing cross-referenced data..."):
        recommendations = engine.get_recommendations(selected_book, n=num_recommendations)
    
    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.subheader(f"Recommended for fans of '{selected_book}':")
        for _, row in recommendations.iterrows():
            with st.container():
                # Displaying as clean cards
                score_percentage = f"{int(row['score'] * 100)}%"
                st.markdown(f"""
                    <div class="rec-card">
                        <span style="float: right; color: #1e3a8a; font-weight: bold;">{score_percentage} Match</span>
                        <h4 style="margin: 0;">{row['name']}</h4>
                        <p style="margin: 0; color: #64748b;">By {row['author']}</p>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.info("👈 Please select a book from the sidebar to view recommendations.")