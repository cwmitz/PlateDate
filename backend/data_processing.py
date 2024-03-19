import re

import numpy as np


def tokenize(text):
    """
    Tokenizes the input string into a list of words and removes any punctuation.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list: The list of words from the input string with any punctuation removed.
    """
    return re.findall(r"[a-z]+", text.lower())


def build_inverted_index(recipes):
    """
    Builds an inverted index from the input list of recipes.

    Args:
        recipes (list): The list of recipes to build the inverted index from.

    Returns:
        dict: The inverted index of the input recipes.
    """
    inverted_index = {}

    for recipe in recipes:
        recipe_id = int(recipe["RecipeId"])

        tokens = set(
            tokenize(recipe["Name"])
            + tokenize(recipe["RecipeIngredientParts"][1:])
            + tokenize(recipe["Keywords"][1:])
            + tokenize(recipe["RecipeCategory"])
        )

        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []

            inverted_index[token].append(recipe_id)

    return inverted_index


def build_id_to_recipe(recipes):
    """
    Builds a dictionary from recipe ID to the parts of the recipe we care about.

    Args:
        recipes (list): The list of recipes to build the dictionary from.

    Returns:
        dict: The dictionary from recipe ID to the parts of the recipe we care about.
    """
    id_to_recipe = {}

    for recipe in recipes:
        recipe_id = int(recipe["RecipeId"])
        id_to_recipe[recipe_id] = {
            "name": recipe["Name"],
            "description": recipe["Description"],
            "allergy/diet": {
                "dairy": False,
                "egg": False,
                "tree_nut": False,
                "peanut": False,
                "shellfish": False,
                "wheat": False,
                "soy": False,
                "fish": False,
                "sesame": False,
                "gluten": False,
                "not_vegetarian": False,
                "not_vegan": False,
            },
            "cook_time": recipe["CookTime"],  # PT format
            "prep_time": recipe["PrepTime"],  # PT format
            "total_time": recipe["TotalTime"],  # PT format
            "review_count": (
                int(recipe["ReviewCount"]) if recipe["ReviewCount"] != "NA" else 0
            ),
            # Set to None if no ratings
            "aggregated_rating": (
                float(recipe["AggregatedRating"])
                if recipe["AggregatedRating"] != "NA"
                else None
            ),
            "ingredients": {},  # ingredient -> quantity
            "instructions": [],  # list of strings
            "yield": recipe["RecipeYield"],
            "servings": recipe["RecipeServings"],
        }

        # Add allergy/diet information

        # Add ingredients

        # Add instructions

    return id_to_recipe


def build_idf(inverted_index, num_recipes):
    """
    Builds the inverse document frequency (IDF) for each term in the input inverted index.

    Args:
        inverted_index (dict): The inverted index to build the IDF from.
        num_recipes (int): The total number of recipes in the dataset.

    Returns:
        dict: The IDF for each term in the input inverted index.
    """
    idf = {}

    for term, postings in inverted_index.items():
        idf[term] = np.log2(num_recipes / (len(postings) + 1))

    return idf


def build_recipe_norms(inverted_index, idf, num_recipes):
    """
    Precomputes the euclidean norm for each recipe.

    Args:
        inverted_index (dict): The inverted index of the dataset.
        idf (dict): The IDF of the dataset.
        num_recipes (int): The total number of recipes in the dataset.

    Returns:
        dict: The euclidean norm of each recipe in the dataset.
    """
    recipe_norms = {}

    for term, postings in inverted_index.items():
        idf_value = idf[term]

        for posting in postings:
            if posting not in recipe_norms:
                recipe_norms[posting] = 0

            recipe_norms[posting] += (idf_value**2) * (
                1 + np.log2(num_recipes / len(postings))
            )

    for recipe_id, norm in recipe_norms.items():
        recipe_norms[recipe_id] = np.sqrt(norm)

    return recipe_norms
