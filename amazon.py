import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Function to preprocess the dataset
@st.cache_data
def load_and_preprocess_data(file):
    # Load the dataset
    if file is None:
        st.error("Please upload the dataset file (ecommerce_dataset.csv).")
        return None, None, None, None
    
    data = pd.read_csv(file)

    # Clean the rating column
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')  # Convert to float, invalid values become NaN

    # Drop rows with invalid ratings
    data = data.dropna(subset=['rating'])

    # Expand the dataset to have one row per user-product interaction
    expanded_rows = []
    for idx, row in data.iterrows():
        user_ids = row['user_id'].split(',')
        user_names = row['user_name'].split(',')
        review_ids = row['review_id'].split(',')
        review_titles = row['review_title'].split(',')
        review_contents = row['review_content'].split(',')
        
        # Ensure all lists have the same length by padding with empty strings if necessary
        max_len = max(len(user_ids), len(user_names), len(review_ids), len(review_titles), len(review_contents))
        user_ids += [''] * (max_len - len(user_ids))
        user_names += [''] * (max_len - len(user_names))
        review_ids += [''] * (max_len - len(review_ids))
        review_titles += [''] * (max_len - len(review_titles))
        review_contents += [''] * (max_len - len(review_contents))
        
        # Create a row for each user
        for i in range(max_len):
            expanded_rows.append({
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'category': row['category'],
                'discounted_price': row['discounted_price'],
                'actual_price': row['actual_price'],
                'discount_percentage': row['discount_percentage'],
                'rating': row['rating'],
                'rating_count': row['rating_count'].replace(',', '') if isinstance(row['rating_count'], str) else row['rating_count'],
                'about_product': row['about_product'],
                'user_id': user_ids[i],
                'user_name': user_names[i],
                'review_id': review_ids[i],
                'review_title': review_titles[i],
                'review_content': review_contents[i],
                'img_link': row['img_link'],
                'product_link': row['product_link']
            })

    # Create a new DataFrame from the expanded rows
    expanded_data = pd.DataFrame(expanded_rows)

    # Drop rows with missing user_id or product_id
    expanded_data = expanded_data[expanded_data['user_id'] != '']
    expanded_data = expanded_data.dropna(subset=['product_id', 'user_id', 'rating'])

    # Remove duplicates by keeping the first occurrence
    expanded_data = expanded_data.drop_duplicates(subset=['user_id', 'product_id'], keep='first')

    # Clean price columns (remove ₹ and convert to float)
    expanded_data['discounted_price'] = expanded_data['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
    expanded_data['actual_price'] = expanded_data['actual_price'].replace('[₹,]', '', regex=True).astype(float)

    # Fill missing values in text columns
    expanded_data['about_product'] = expanded_data['about_product'].fillna('')
    expanded_data['review_content'] = expanded_data['review_content'].fillna('')
    expanded_data['category'] = expanded_data['category'].fillna('')

    # Combine text features for content-based filtering
    expanded_data['combined_features'] = expanded_data['category'] + ' ' + expanded_data['about_product'] + ' ' + expanded_data['review_content']

    # Collaborative Filtering: Create user-item matrix
    user_item_matrix = expanded_data.pivot(index='user_id', columns='product_id', values='rating').fillna(0)

    # Compute user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Map product_id to product_name for display
    pid_to_name = expanded_data.drop_duplicates(subset=['product_id']).set_index('product_id')['product_name'].to_dict()

    # Content-Based Filtering: Create TF-IDF matrix for content features
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(expanded_data['combined_features'])

    # Compute item similarity
    item_similarity = cosine_similarity(tfidf_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=expanded_data.index, columns=expanded_data.index)

    return expanded_data, user_item_matrix, user_similarity_df, item_similarity_df, pid_to_name

# Collaborative Filtering Recommendation
def collaborative_filtering_recommendation(user, user_item_matrix, user_similarity_df, pid_to_name, top_n=2):
    if user not in user_similarity_df.index:
        st.warning(f"User {user} not found in the dataset.")
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])
    
    # Get similar users
    similar_users = user_similarity_df[user].sort_values(ascending=False)[1:]  # Exclude the user itself
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    
    # Compute weighted ratings
    weighted_ratings = np.dot(similar_users.values, similar_users_ratings) / (similar_users.sum() + 1e-8)
    predictions = pd.Series(weighted_ratings, index=user_item_matrix.columns)
    
    # Filter out items the user has already rated
    user_rated = user_item_matrix.loc[user]
    unrated_items = predictions[user_rated == 0].sort_values(ascending=False)
    
    # Get top N recommendations
    recommended_pids = unrated_items.head(top_n).index
    recommended_names = [pid_to_name.get(pid, "Unknown Product") for pid in recommended_pids]
    return pd.DataFrame({
        'product_id': recommended_pids,
        'product_name': recommended_names,
        'score': unrated_items.head(top_n).values
    })

# Content-Based Recommendation
def content_based_recommendation(item_pid, preferences, expanded_data, item_similarity_df, pid_to_name, top_n=2):
    # Filter items based on preferences (e.g., category and price range)
    filtered_df = expanded_data[
        (expanded_data['category'].str.contains(preferences['category'], case=False, na=False)) &
        (expanded_data['discounted_price'] >= preferences['min_price']) &
        (expanded_data['discounted_price'] <= preferences['max_price'])
    ]
    if filtered_df.empty:
        st.warning("No items match the preferences.")
        return pd.DataFrame(columns=['product_id', 'product_name', 'score'])
    
    # Find the index of the item_pid
    item_idx = expanded_data[expanded_data['product_id'] == item_pid].index
    if not item_idx.empty:
        item_idx = item_idx[0]
    else:
        st.warning(f"Item {item_pid} not found.")
        return filtered_df.sort_values(by='rating', ascending=False).head(top_n)[['product_id', 'product_name', 'rating']]
    
    # Get similar items
    similar_items = item_similarity_df.iloc[item_idx].loc[filtered_df.index].sort_values(ascending=False)
    top_similar_indices = similar_items.head(top_n + 1).index[1:]  # Exclude the item itself
    
    # Get recommendations
    recommendations = filtered_df.loc[top_similar_indices][['product_id', 'product_name', 'rating']]
    recommendations['score'] = similar_items.loc[top_similar_indices].values
    return recommendations[['product_id', 'product_name', 'score']].head(top_n)

# Hybrid Recommendation
def hybrid_recommendation(user, liked_pid, preferences, expanded_data, user_item_matrix, user_similarity_df, item_similarity_df, pid_to_name, top_n=2):
    # Get collaborative filtering recommendations
    collab_preds = collaborative_filtering_recommendation(user, user_item_matrix, user_similarity_df, pid_to_name, top_n=5)
    
    # Get content-based recommendations
    content_recs = content_based_recommendation(liked_pid, preferences, expanded_data, item_similarity_df, pid_to_name, top_n=5)
    
    if content_recs.empty:
        return collab_preds.head(top_n)
    
    # Combine scores (weighted combination)
    content_scores = pd.Series(content_recs['score'].values, index=content_recs['product_id']).to_dict()
    collab_scores = pd.Series(collab_preds['score'].values, index=collab_preds['product_id']).to_dict()
    
    hybrid_scores = {}
    for pid in set(collab_scores.keys()).union(content_scores.keys()):
        collab_score = collab_scores.get(pid, 0)
        content_score = content_scores.get(pid, 0)
        hybrid_scores[pid] = 0.4 * content_score + 0.6 * collab_score  # Prioritize CF
    
    # Get top N hybrid recommendations
    top_pids = pd.Series(hybrid_scores).sort_values(ascending=False).head(top_n).index
    return pd.DataFrame({
        'product_id': top_pids,
        'product_name': [pid_to_name.get(pid, "Unknown Product") for pid in top_pids],
        'score': pd.Series(hybrid_scores).loc[top_pids].values
    })

# Streamlit App
def main():
    st.title("E-Commerce Recommendation System")

    # File uploader for the dataset
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload ecommerce_dataset.csv", type=["csv"])

    # Load and preprocess data
    if uploaded_file is not None:
        expanded_data, user_item_matrix, user_similarity_df, item_similarity_df, pid_to_name = load_and_preprocess_data(uploaded_file)
        
        if expanded_data is None:
            return

        # Sidebar: Recommendation Settings
        st.sidebar.header("Recommendation Settings")
        recommendation_type = st.sidebar.selectbox(
            "Select Recommendation Type",
            ["Collaborative Filtering", "Content-Based Filtering", "Hybrid Recommendation"]
        )
        top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=2)

        # User selection for Collaborative Filtering and Hybrid
        if recommendation_type in ["Collaborative Filtering", "Hybrid Recommendation"]:
            st.sidebar.header("User Selection")
            users = expanded_data['user_id'].drop_duplicates().tolist()
            selected_user = st.sidebar.selectbox("Select a User", users)

        # Product selection for Content-Based Filtering and Hybrid
        if recommendation_type in ["Content-Based Filtering", "Hybrid Recommendation"]:
            st.sidebar.header("Product Selection")
            products = expanded_data[['product_id', 'product_name']].drop_duplicates()
            product_options = [f"{row['product_id']} - {row['product_name']}" for _, row in products.iterrows()]
            selected_product = st.sidebar.selectbox("Select a Product", product_options)
            selected_pid = selected_product.split(" - ")[0]

            # Preferences for Content-Based Filtering
            st.sidebar.header("Preferences")
            category = st.sidebar.text_input("Category (e.g., USBCables, leave blank for no filter)", "")
            min_price = st.sidebar.number_input("Minimum Price", min_value=0.0, value=0.0, step=10.0)
            max_price = st.sidebar.number_input("Maximum Price", min_value=0.0, value=float('inf'), step=10.0)
            preferences = {
                'category': category,
                'min_price': min_price,
                'max_price': max_price if max_price != float('inf') else 1e9  # Use a large number for "no limit"
            }

        # Generate Recommendations
        if st.sidebar.button("Generate Recommendations"):
            if recommendation_type == "Collaborative Filtering":
                st.header(f"Collaborative Filtering Recommendations for User {selected_user}")
                recs = collaborative_filtering_recommendation(selected_user, user_item_matrix, user_similarity_df, pid_to_name, top_n=top_n)
                st.dataframe(recs)

            elif recommendation_type == "Content-Based Filtering":
                st.header(f"Content-Based Recommendations for Product {selected_pid}")
                recs = content_based_recommendation(selected_pid, preferences, expanded_data, item_similarity_df, pid_to_name, top_n=top_n)
                st.dataframe(recs)

            elif recommendation_type == "Hybrid Recommendation":
                st.header(f"Hybrid Recommendations for User {selected_user} and Product {selected_pid}")
                recs = hybrid_recommendation(selected_user, selected_pid, preferences, expanded_data, user_item_matrix, user_similarity_df, item_similarity_df, pid_to_name, top_n=top_n)
                st.dataframe(recs)

                # Evaluation
                user_rated = user_item_matrix.loc[selected_user]
                ground_truth = user_rated[user_rated >= 4].sort_values(ascending=False).index.tolist()[:3]
                recommendations = recs['product_id'].tolist()

                st.subheader("Debugging Information")
                st.write("Ground Truth (Top 3 product_ids rated by user):", ground_truth)
                st.write("Hybrid Recommendations (product_ids):", recommendations)

                # Compute precision, recall, and F1 score
                true_positives = len(set(ground_truth).intersection(recommendations))
                precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
                recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                st.subheader("Evaluation Metrics")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Compute RMSE for collaborative filtering predictions
                collab_preds = collaborative_filtering_recommendation(selected_user, user_item_matrix, user_similarity_df, pid_to_name, top_n=top_n)
                actual_ratings = user_rated[user_rated > 0].tolist()[:2]
                predicted_ratings = [collab_preds.set_index('product_id')['score'].get(pid, 0) for pid in user_rated[user_rated > 0].index[:2]]
                rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
                st.write(f"RMSE: {rmse:.2f}")
    else:
        st.info("Please upload the dataset to start the recommendation system.")

if __name__ == "__main__":
    main()