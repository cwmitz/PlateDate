from sklearn.metrics.pairwise import cosine_similarity


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


def algorithm(
    queries,
    dietary_restrictions,
    id_to_recipe,
    vectorizer,
    svd_model,
    svd,
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

    def cos_sim(query):
        query_vec = vectorizer.transform([query])
        query_svd = svd_model.transform(query_vec)

        sim_scores = cosine_similarity(query_svd, svd)

        ids = list(id_to_recipe.keys())

        results = [(ids[i], sim_scores[0][i]) for i in range(len(sim_scores[0]))]
        return results

    # Run cosine similarity for each query
    cosine_scores = [cos_sim(query) for query in queries]

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
