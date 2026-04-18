# models.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
from surprise import accuracy


#COSINE MODEL 

def compute_item_similarity(user_item_matrix):
    # Fill missing ratings with 0 so that similarity can be computed properly
    matrix_filled = user_item_matrix.fillna(0)

    # Transpose the matrix so that items become rows and users become columns
    item_matrix = matrix_filled.T

    # Compute cosine similarity between all items
    similarity_matrix = cosine_similarity(item_matrix)

    return similarity_matrix, item_matrix


def recommend_items(user_id, user_item_matrix, similarity_matrix, item_matrix, top_k=5):
    # Fetch the ratings given by the selected user
    user_ratings = user_item_matrix.loc[user_id].fillna(0)

    # Compute a score for each item based on similarity and user preferences
    scores = similarity_matrix.dot(user_ratings)

    # Identify items that the user has already rated
    rated_items = user_ratings[user_ratings > 0].index

    # Sort items in descending order of their scores
    ranked_items = np.argsort(scores)[::-1]

    recommendations = []

    for idx in ranked_items:
        item_id = item_matrix.index[idx]

        # Skip items that the user has already interacted with
        if item_id not in rated_items:
            recommendations.append(int(item_id))

        # Stop once we have enough recommendations
        if len(recommendations) >= top_k:
            break

    return recommendations


#SVD MODEL 

def train_svd_model(data):
    # Define the rating scale used in the dataset
    reader = Reader(rating_scale=(0.5, 5.0))

    # Convert the pandas dataframe into a Surprise dataset format
    surprise_data = Dataset.load_from_df(
        data[['user_id', 'item_id', 'rating']],
        reader
    )

    # Split the data into training and testing sets for proper evaluation
    trainset, testset = surprise_split(
        surprise_data,
        test_size=0.2,
        random_state=42
    )

    # Initialize the SVD model with chosen hyperparameters
    model = SVD(n_factors=20, n_epochs=10)

    # Train the model on the training set
    model.fit(trainset)

    # Generate predictions on the test set
    predictions = model.test(testset)

    # Compute RMSE to evaluate prediction accuracy
    rmse = accuracy.rmse(predictions)

    return model, rmse


def recommend_svd(user_id, data, model, top_k=5):
    # Retrieve all unique items from the dataset
    all_items = data['item_id'].unique()

    # Find items that the user has already rated
    rated_items = data[data['user_id'] == user_id]['item_id'].tolist()

    predictions = []

    for item in all_items:
        # Only consider items that the user has not seen before
        if item not in rated_items:
            pred = model.predict(user_id, item)
            predictions.append((item, pred.est))

    # Sort items based on predicted rating in descending order
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Select the top_k items as recommendations
    top_items = [int(item[0]) for item in predictions[:top_k]]

    return top_items