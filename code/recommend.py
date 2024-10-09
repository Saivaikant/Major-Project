# recommend.py
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import os

# Step 1: Load the output.txt data into a user-item interaction matrix
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Split each line into user ID and item IDs
    data = []
    for line in lines:
        line = line.strip().split()
        user_id = int(line[0])
        item_ids = [int(item) for item in line[1:]]
        for item_id in item_ids:
            data.append([user_id, item_id, 1])  # 1 indicates interaction
    
    return pd.DataFrame(data, columns=['user_id', 'item_id', 'interaction'])

# Load the interaction data from your output file
def create_interaction_matrix(file_path):
    interaction_df = load_data(file_path)
    interaction_matrix = interaction_df.pivot_table(index='user_id', columns='item_id', values='interaction', fill_value=0)
    return interaction_matrix

# Step 2: Apply Matrix Factorization using SVD
def apply_svd(matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)  # Decompose into user factors
    item_factors = svd.components_  # Decompose into item factors
    return user_factors, item_factors

# Step 3: Generate Recommendations for a user
def recommend_items(user_id, user_factors, item_factors, interaction_matrix, top_n=5):
    user_idx = interaction_matrix.index.get_loc(user_id)  # Find the row index of the user
    user_latent_factors = user_factors[user_idx, :]
    
    # Calculate the predicted interaction scores for all items (dot product)
    predicted_scores = np.dot(user_latent_factors, item_factors)
    
    # Get the items that the user hasn't interacted with
    interacted_items = set(interaction_matrix.columns[interaction_matrix.iloc[user_idx] > 0])
    all_items = set(interaction_matrix.columns)
    non_interacted_items = all_items - interacted_items
    
    # Rank the non-interacted items by predicted score
    non_interacted_items_scores = {item: predicted_scores[interaction_matrix.columns.get_loc(item)]
                                   for item in non_interacted_items}
    recommended_items = sorted(non_interacted_items_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [item for item, score in recommended_items]

# Main function to generate recommendations
def generate_recommendations(file_path, user_id, top_n=5):
    interaction_matrix = create_interaction_matrix(file_path)
    user_factors, item_factors = apply_svd(interaction_matrix)
    recommendations = recommend_items(user_id, user_factors, item_factors, interaction_matrix, top_n)
    
    return recommendations
