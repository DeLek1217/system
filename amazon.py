import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Amazon Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    .stButton button {
        width: 100%;
        background-color: #FF9900;
        color: white;
    }
    .stButton button:hover {
        background-color: #FF7700;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset with optimizations"""
    data = pd.read_csv('amazon.csv')
    
    # Clean rating column
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating'])
    
    # For development, use a subset of data (remove in production)
    if st.session_state.get('dev_mode', True):
        data = data.sample(frac=0.3, random_state=42)
    
    # Expand the dataset
    expanded_rows = []
    for idx, row in data.iterrows():
        user_ids = row['user_id'].split(',')
        user_names = row['user_name'].split(',')
        review_ids = row['review_id'].split(',')
        review_titles = row['review_title'].split(',')
        review_contents = row['review_content'].split(',')
        
        max_len = max(len(user_ids), len(user_names), len(review_ids), 
                    len(review_titles), len(review_contents))
        user_ids += [''] * (max_len - len(user_ids))
        user_names += [''] * (max_len - len(user_names))
        review_ids += [''] * (max_len - len(review_ids))
        review_titles += [''] * (max_len - len(review_titles))
        review_contents += [''] * (max_len - len(review_contents))
        
        for i in range(max_len):
            expanded_rows.append({
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'category': row['category'],
                'discounted_price': float(str(row['discounted_price']).replace('₹','').replace(',','')),
                'actual_price': float(str(row['actual_price']).replace('₹','').replace(',','')),
                'discount_percentage': row['discount_percentage'],
                'rating': row['rating'],
                'rating_count': row['rating_count'],
                'about_product': row['about_product'],
                'user_id': user_ids[i],
                'user_name': user_names[i],
                'review_id': review_ids[i],
                'review_title': review_titles[i],
                'review_content': review_contents[i],
                'img_link': row['img_link'],
                'product_link': row['product_link']
            })

    expanded_data = pd.DataFrame(expanded_rows)
    expanded_data = expanded_data[expanded_data['user_id'] != '']
    expanded_data = expanded_data.dropna(subset=['product_id', 'user_id', 'rating'])
    expanded_data = expanded_data.drop_duplicates(subset=['user_id', 'product_id'], keep='first')
    
    # Combine text features
    expanded_data['combined_features'] = (
        expanded_data['category'].fillna('') + ' ' + 
        expanded_data['about_product'].fillna('') + ' ' + 
        expanded_data['review_content'].fillna('')
    )
    
    return expanded_data

@st.cache_resource
def initialize_models(_expanded_data):
    """Initialize all models and similarity matrices"""
    with st.spinner('Initializing recommendation models...'):
        # User-Item Matrix
        user_item_matrix = _expanded_data.pivot(
            index='user_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        # User Similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        # Content-Based Features
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(_expanded_data['combined_features'])
        
        # Item Similarity
        item_similarity = cosine_similarity(tfidf_matrix)
        item_similarity_df = pd.DataFrame(
            item_similarity,
            index=_expanded_data.index,
            columns=_expanded_data.index
        )
        
        # Product ID to name mapping
        pid_to_name = _expanded_data.drop_duplicates(subset=['product_id']).set_index('product_id')['product_name'].to_dict()
        
        return {
            'user_item_matrix': user_item_matrix,
            'user_similarity_df': user_similarity_df,
            'item_similarity_df': item_similarity_df,
            'pid_to_name': pid_to_name,
            'tfidf_matrix': tfidf_matrix,
            'expanded_data': _expanded_data
        }

def collaborative_filtering_recommendation(user_id, top_n=5):
    """Optimized collaborative filtering recommendations"""
    if user_id not in st.session_state.models['user_similarity_df'].index:
        return pd.DataFrame()
    
    # Get similar users
    similar_users = st.session_state.models['user_similarity_df'][user_id].sort_values(ascending=False)[1:11]  # Top 10 similar users
    
    # Get their ratings
    similar_users_ratings = st.session_state.models['user_item_matrix'].loc[similar_users.index]
    
    # Compute weighted average
    weighted_ratings = np.dot(similar_users.values, similar_users_ratings) / (similar_users.sum() + 1e-8)
    predictions = pd.Series(weighted_ratings, index=st.session_state.models['user_item_matrix'].columns)
    
    # Filter out items the user has already rated
    user_rated = st.session_state.models['user_item_matrix'].loc[user_id]
    unrated_items = predictions[user_rated == 0].sort_values(ascending=False)
    
    # Get top N recommendations
    recommended_pids = unrated_items.head(top_n).index
    recommended_names = [st.session_state.models['pid_to_name'].get(pid, "Unknown Product") for pid in recommended_pids]
    
    return pd.DataFrame({
        'product_id': recommended_pids,
        'product_name': recommended_names,
        'score': unrated_items.head(top_n).values
    })

def content_based_recommendation(product_id, preferences, top_n=5):
    """Optimized content-based recommendations with preferences"""
    # Filter items based on preferences
    filtered_data = st.session_state.models['expanded_data'][
        (st.session_state.models['expanded_data']['category'].str.contains(
            preferences['category'], case=False, na=False)) &
        (st.session_state.models['expanded_data']['discounted_price'] >= preferences['min_price']) &
        (st.session_state.models['expanded_data']['discounted_price'] <= preferences['max_price'])
    ]
    
    if filtered_data.empty:
        return pd.DataFrame()
    
    # Find the index of the product
    product_idx = filtered_data[filtered_data['product_id'] == product_id].index
    if product_idx.empty:
        return filtered_data.sort_values(by='rating', ascending=False).head(top_n)[['product_id', 'product_name', 'rating']]
    
    product_idx = product_idx[0]
    
    # Get similar items from the filtered subset
    similar_items = st.session_state.models['item_similarity_df'].iloc[product_idx].loc[filtered_data.index].sort_values(ascending=False)
    top_indices = similar_items.head(top_n + 1).index[1:]  # Exclude self
    
    recommendations = filtered_data.loc[top_indices][['product_id', 'product_name', 'rating']]
    recommendations['score'] = similar_items.loc[top_indices].values
    
    return recommendations.head(top_n)

def hybrid_recommendation(user_id, product_id, preferences, top_n=5):
    """Optimized hybrid recommendations with parallel execution"""
    with ThreadPoolExecutor() as executor:
        # Run both recommendation systems in parallel
        cf_future = executor.submit(collaborative_filtering_recommendation, user_id, top_n*2)
        cb_future = executor.submit(content_based_recommendation, product_id, preferences, top_n*2)
        
        cf_recs = cf_future.result()
        cb_recs = cb_future.result()
    
    if cb_recs.empty:
        return cf_recs.head(top_n)
    
    # Combine scores
    content_scores = pd.Series(cb_recs['score'].values, index=cb_recs['product_id']).to_dict()
    collab_scores = pd.Series(cf_recs['score'].values, index=cf_recs['product_id']).to_dict()
    
    hybrid_scores = {}
    for pid in set(collab_scores.keys()).union(content_scores.keys()):
        hybrid_scores[pid] = (0.4 * content_scores.get(pid, 0) + 
                             0.6 * collab_scores.get(pid, 0))
    
    # Get top N hybrid recommendations
    top_pids = pd.Series(hybrid_scores).sort_values(ascending=False).head(top_n).index
    return pd.DataFrame({
        'product_id': top_pids,
        'product_name': [st.session_state.models['pid_to_name'].get(pid, "Unknown Product") for pid in top_pids],
        'score': pd.Series(hybrid_scores).loc[top_pids].values
    })

def display_product_card(row):
    """Display product information in a card format"""
    with st.container():
        st.markdown(f"""
        <div class="product-card">
            <h3>{row['product_name']}</h3>
            <p><strong>Rating:</strong> {row['rating']} ⭐</p>
            <p><strong>Price:</strong> ₹{row['discounted_price']:,.2f} 
            <small><s>₹{row['actual_price']:,.2f}</s></small></p>
            <p><strong>Category:</strong> {row['category']}</p>
            <p>{row['about_product'][:150]}...</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    st.title("🛍️ Amazon E-commerce Recommendation System")
    st.markdown("Discover products tailored just for you!")
    
    # Initialize data and models
    if 'models' not in st.session_state:
        data = load_and_preprocess_data()
        st.session_state.models = initialize_models(data)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Recommendation Options")
        rec_type = st.radio(
            "Recommendation Type:",
            ["Collaborative Filtering", "Content-Based", "Hybrid"],
            index=2
        )
        
        top_n = st.slider("Number of Recommendations:", 1, 10, 5)
        
        if rec_type != "Collaborative Filtering":
            st.subheader("Content Filters")
            category = st.text_input("Product Category (e.g., Electronics):", "")
            price_range = st.slider(
                "Price Range (₹):",
                0, 10000, (500, 5000)
            )
    
    # Main content area
    tab1, tab2 = st.tabs(["Recommendations", "Product Explorer"])
    
    with tab1:
        if rec_type == "Collaborative Filtering":
            st.header("Collaborative Filtering Recommendations")
            user_id = st.selectbox(
                "Select User:",
                st.session_state.models['user_item_matrix'].index.unique()[:20],
                index=0
            )
            
            if st.button("Get Recommendations", key="cf_rec"):
                with st.spinner("Finding similar users..."):
                    recs = collaborative_filtering_recommendation(user_id, top_n)
                
                if not recs.empty:
                    st.success(f"Top {len(recs)} recommendations for user {user_id}")
                    for _, row in recs.merge(
                        st.session_state.models['expanded_data'].drop_duplicates('product_id'), 
                        on='product_id'
                    ).iterrows():
                        display_product_card(row)
                else:
                    st.warning("No recommendations found for this user.")
        
        elif rec_type == "Content-Based":
            st.header("Content-Based Recommendations")
            product_id = st.selectbox(
                "Select a Product You Like:",
                st.session_state.models['expanded_data'][['product_id', 'product_name']]
                    .drop_duplicates()['product_name'].head(50),
                index=0
            )
            
            if st.button("Get Recommendations", key="cb_rec"):
                with st.spinner("Finding similar products..."):
                    pid = st.session_state.models['expanded_data'][
                        st.session_state.models['expanded_data']['product_name'] == product_id
                    ]['product_id'].iloc[0]
                    
                    preferences = {
                        'category': category,
                        'min_price': price_range[0],
                        'max_price': price_range[1]
                    }
                    
                    recs = content_based_recommendation(pid, preferences, top_n)
                
                if not recs.empty:
                    st.success(f"Products similar to {product_id}")
                    for _, row in recs.merge(
                        st.session_state.models['expanded_data'].drop_duplicates('product_id'), 
                        on='product_id'
                    ).iterrows():
                        display_product_card(row)
                else:
                    st.warning("No similar products found matching your criteria.")
        
        else:  # Hybrid
            st.header("Hybrid Recommendations")
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.selectbox(
                    "Select User:",
                    st.session_state.models['user_item_matrix'].index.unique()[:20],
                    index=0
                )
            
            with col2:
                product_id = st.selectbox(
                    "Select a Product You Like:",
                    st.session_state.models['expanded_data'][['product_id', 'product_name']]
                        .drop_duplicates()['product_name'].head(50),
                    index=0
                )
            
            if st.button("Get Recommendations", key="hybrid_rec"):
                with st.spinner("Analyzing your preferences..."):
                    pid = st.session_state.models['expanded_data'][
                        st.session_state.models['expanded_data']['product_name'] == product_id
                    ]['product_id'].iloc[0]
                    
                    preferences = {
                        'category': category,
                        'min_price': price_range[0],
                        'max_price': price_range[1]
                    }
                    
                    recs = hybrid_recommendation(user_id, pid, preferences, top_n)
                
                if not recs.empty:
                    st.success(f"Personalized recommendations for user {user_id}")
                    for _, row in recs.merge(
                        st.session_state.models['expanded_data'].drop_duplicates('product_id'), 
                        on='product_id'
                    ).iterrows():
                        display_product_card(row)
                    
                    # Evaluation metrics
                    with st.expander("Show Recommendation Metrics"):
                        user_rated = st.session_state.models['user_item_matrix'].loc[user_id]
                        ground_truth = user_rated[user_rated >= 4].sort_values(ascending=False).index.tolist()[:3]
                        recommendations = recs['product_id'].tolist()
                        
                        true_positives = len(set(ground_truth).intersection(recommendations))
                        precision = true_positives / len(recommendations) if recommendations else 0
                        recall = true_positives / len(ground_truth) if ground_truth else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                        st.metric("F1 Score", f"{f1:.2%}")
                else:
                    st.warning("No recommendations found matching your criteria.")
    
    with tab2:
        st.header("Product Explorer")
        search_term = st.text_input("Search products:")
        
        if search_term:
            results = st.session_state.models['expanded_data'][
                st.session_state.models['expanded_data']['product_name'].str.contains(
                    search_term, case=False)
            ].drop_duplicates('product_id').head(20)
            
            if not results.empty:
                for _, row in results.iterrows():
                    display_product_card(row)
            else:
                st.warning("No products found matching your search.")

if __name__ == "__main__":
    main()
