# data_preprocessing.py
# Importing required functions for Spark-based processing
from spark_processing import create_spark_session, load_data_spark, preprocess_data_spark, convert_to_pandas

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, use_spark=True):
    """
    This function loads the dataset.

    If use_spark is True, data is processed using Spark (useful for large datasets).
    Otherwise, it falls back to pandas for simpler loading.
    """
    if use_spark:
        # Create a Spark session for distributed data processing
        spark = create_spark_session()

        # Load the dataset into a Spark DataFrame
        df_spark = load_data_spark(spark, path)

        # Perform preprocessing using Spark 
        df_spark = preprocess_data_spark(df_spark)

        # Convert the Spark DataFrame into a pandas DataFrame
        data = convert_to_pandas(df_spark)

    else:
        # If Spark is not used, directly read the CSV using pandas
        data = pd.read_csv(path)

    return data


def preprocess_data(data):
    """
    Performs basic preprocessing on the dataset:
    - Renames columns for consistency
    - Keeps only relevant columns needed for recommendation
    """

    # Renaming columns to a standard format 
    data = data.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id'
    })

    # Selecting only the important columns
    data = data[['user_id', 'item_id', 'rating']]

    return data


def create_user_item_matrix(data):
    """
    Converts the data into a user-item matrix.
    Rows represent users, columns represent items (movies) and values represent the ratings given by users.
    
    """

    user_item_matrix = data.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    )

    return user_item_matrix


def train_test_split_data(data):
    """
    Splits the dataset into training and testing sets.
    Training data is used to build the recommendation models.
    Testing data is used to evaluate their performance.
    """

    train, test = train_test_split(
        data,
        test_size=0.2,   # 80% training, 20% testing
        random_state=42  
    )

    return train, test