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
                dot_scores[int(recipe_id)] = (
                    dot_scores.get(int(recipe_id), 0) + idf[term]
                )

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
        cosine_scores (list): List of (recipe ID, cosine similarity score) pairs.
    """
    # Tokenize the input query
    query_set = data_processing.tokenize(query, inverted_index)

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
            (int(recipe_id), dot_score / (query_norm * recipe_norms[int(recipe_id)]))
        )

    return cosine_scores


def common_recipes(cosine_scores_all):
    """
    Gets the top 10 most common recipes from the cosine similarity scores of
    all queries.

    Args:
        cosine_scores_all (list): List of cosine similarity scores for each query.

    Returns:
        top_10_recipes (list): A list of tuples of (recipe_id, score)
    """
    # Dictionary to store accumulated ranks of each recipe
    accumulated_ranks = {}

    # Accumulate ranks of each recipe
    for cosine_scores in cosine_scores_all:
        for recipe_id, score in cosine_scores:
            if recipe_id in accumulated_ranks:
                accumulated_ranks[recipe_id] += score
            else:
                accumulated_ranks[recipe_id] = score
    # Sort the accumulated ranks in increasing order
    top_recipes = sorted(accumulated_ranks.items(), key=lambda x: x[1], reverse=True)

    return top_recipes


def get_sim_scores(top_recipes, cosine_scores_all, num_queries):
    """
    Modifies the scores dictionary to include a score for every query for each recipe.
    If a recipe does not have a score for a particular query, it is assumed to be zero.
    """
    # Get recipe IDs from top recipes
    top_ids = {recipe_id for recipe_id, _ in top_recipes}
    # Initialize dictionary to store scores with a list containing zeros initially for each query
    scores = {recipe_id: [0] * num_queries for recipe_id in top_ids}

    for query_index, s in enumerate(cosine_scores_all):
        for recipe_id, score in s:
            if recipe_id in top_ids:
                # Replace the zero at the query index with the actual score
                scores[recipe_id][query_index] = score

    return scores


# def iso_to_min(iso_time):
#     """
#     Converts ISO 8601 formatted time to minutes.

#     Args:
#         iso_time (str): The time in ISO 8601 format.

#     Returns:
#         int: The time in minutes.
#     """
#     if iso_to_min == "PT0S":
#         return 0
#     total_minutes = 0
#     iso_time = iso_time[2:]
#     if "H" in iso_time:
#         total_minutes += int(iso_time.split("H")[0]) * 60
#         h_idx = iso_time.index("H")
#         iso_time = iso_time[h_idx + 1 :]
#     if "M" in iso_time:
#         total_minutes += int(iso_time.split("M")[0])
#     return total_minutes


def algorithm(
    queries,
    dietary_restrictions,
    inverted_index,
    idf,
    recipe_norms,
    id_to_recipe,
    time_limit,
):
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

    top_recipes = common_recipes(cosine_scores)
    # Filter out recipes based on dietary restrictions
    top_recipes_filtered = []
    for recipe_id, _ in top_recipes:
        recipe = id_to_recipe[str(recipe_id)]
        # These two lines break the code
        # if iso_to_min(recipe["total_time"]) > time_limit:
        #     continue
        if (
            dietary_restrictions["vegetarian"]
            and not recipe["dietary_restrictions"]["vegetarian"]
        ):
            continue
        if (
            dietary_restrictions["vegan"]
            and not recipe["dietary_restrictions"]["vegan"]
        ):
            continue
        if (
            dietary_restrictions["gluten_free"]
            and not recipe["dietary_restrictions"]["gluten_free"]
        ):
            continue
        if (
            dietary_restrictions["dairy_free"]
            and not recipe["dietary_restrictions"]["dairy_free"]
        ):
            continue
        if (
            dietary_restrictions["nut_free"]
            and not recipe["dietary_restrictions"]["nut_free"]
        ):
            continue
        top_recipes_filtered.append((recipe_id, 0))

    num_queries = len(queries)

    # Get the top 10 most common recipes from the cosine similarity scores
    return top_recipes_filtered[:10], get_sim_scores(
        top_recipes, cosine_scores, num_queries
    )
