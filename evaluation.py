# evaluation.py

def precision_at_k(recommended_items, relevant_items, k):
    # Take only the top k recommended items
    recommended_at_k = recommended_items[:k]

    # Convert relevant items to a set for faster lookup
    relevant_set = set(relevant_items)

    hits = 0

    # Count how many recommended items are actually relevant
    for item in recommended_at_k:
        if item in relevant_set:
            hits += 1

    # Precision measures how many of the recommended items were correct
    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    # Take only the top k recommended items
    recommended_at_k = recommended_items[:k]

    # Convert relevant items to a set for faster lookup
    relevant_set = set(relevant_items)

    hits = 0

    # Count how many relevant items were successfully recommended
    for item in recommended_at_k:
        if item in relevant_set:
            hits += 1

    # If there are no relevant items, recall is defined as 0
    if len(relevant_set) == 0:
        return 0

    # Recall measures how many of the actual relevant items were retrieved
    return hits / len(relevant_set)