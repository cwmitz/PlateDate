import data_processing
import numpy as np


def accumulate_dot_scores(query_set, inverted_index, idf):
    """
    Performs a term-at-a-time iteration to efficiently compute the numerator term
    of the cosine similarity formula for each recipe.

    Args:
        query_set (set): The set of terms in the input query.
        inverted_index (dict): The inverted index of the dataset.
        idf (dict): The IDF of the dataset.

    Returns:
        dot_scores (dict): Mapping of recipe ID to the numerator term of the cosine similarity formula.
    """
    dot_scores = {}

    for term in query_set:
        if term in inverted_index:
            for recipe_id in inverted_index[term]:
                dot_scores[int(recipe_id)] = dot_scores.get(
                    int(recipe_id), 0) + idf[term]

    return dot_scores


def cosine_similarity(query, inverted_index, idf, recipe_norms):
    """
    Computes the cosine similarity between the input query and each recipe in the dataset.

    Args:
        query (str): The input query to search for.
        inverted_index (dict): The inverted index of the dataset.
        idf (dict): The IDF of the dataset.
        recipe_norms (dict): The euclidean norm of each recipe in the dataset.

    Returns:
        cosine_scores (list): Sorted (decreasing) list of (recipe ID, cosine similarity score) pairs.
    """
    # Tokenize the input query
    query_set = data_processing.tokenize(query)

    # Compute query norm
    query_norm = 0
    for term in query_set:
        if term in idf:
            query_norm += idf[term] ** 2
    query_norm = np.sqrt(query_norm)

    # Compute the numerator term of the cosine similarity formula for each recipe
    dot_scores = accumulate_dot_scores(query_set, inverted_index, idf)

    # Compute the cosine similarity score for each recipe
    cosine_scores = []
    for recipe_id, dot_score in dot_scores.items():
        cosine_scores.append(
            (int(recipe_id), dot_score /
             (query_norm * recipe_norms[int(recipe_id)]))
        )

    # Sort the cosine similarity scores in decreasing order
    cosine_scores.sort(key=lambda x: x[1], reverse=True)

    return cosine_scores


def common_recipes(cosine_scores_all):
    """
    Gets the top 10 most common recipes from the cosine similarity scores of
    all queries.

    Args:
        cosine_scores_all (list): List of cosine similarity scores for each query.

    Returns:
        top_10_recipes (list): The top 10 most common recipes from the cosine similarity scores.
    """
    # Dictionary to store accumulated ranks of each recipe
    accumulated_ranks = {}

    # Accumulate ranks of each recipe
    for cosine_scores in cosine_scores_all:
        for rank, (recipe_id, _) in enumerate(cosine_scores):
            accumulated_ranks[int(recipe_id)] = accumulated_ranks.get(
                int(recipe_id), 0) + rank

    # Sort the accumulated ranks in increasing order
    top_10_recipes = sorted(accumulated_ranks.items(), key=lambda x: x[1])[:10]

    return top_10_recipes


def algorithm(queries, inverted_index, idf, recipe_norms):
    """
    Runs cosine similarity for each query and then calculates the most common
    recipes from the results.

    Args:
        queries (list): List of input queries to search for.
        inverted_index (dict): The inverted index of the dataset.
        idf (dict): The IDF of the dataset.
        recipe_norms (dict): The euclidean norm of each recipe in the dataset.

    Returns:
        list: The top 10 recipes in common for all queries.
    """
    # Run cosine similarity for each query
    cosine_scores = [
        cosine_similarity(query, inverted_index, idf, recipe_norms) for query in queries
    ]

    # Get the top 10 most common recipes from the cosine similarity scores
    return common_recipes(cosine_scores)
