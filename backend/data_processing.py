# Load the data from the csv files and process them using numpy
import csv
import os
import re

# This file pre-processes the data by creating an inverted index and recipe id to recipe and reviews dictionary.

os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

recipes_file_path = os.path.join(current_directory, "recipes.csv")
reviews_file_path = os.path.join(current_directory, "reviews.csv")

with open(recipes_file_path, "r") as file:
    recipes = list(csv.DictReader(file))

with open(reviews_file_path, "r") as file:
    reviews = list(csv.DictReader(file))


def tokenize(text):
    """
    Tokenizes the text into words and returns a list of words.

    Args:
    text: string

    Returns:
    list of words
    """
    return re.findall(r"[a-z]+", text.lower())


def build_inverted_index(recipes):
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


def build_id_to_recipe(recipes, reviews):
    """
    This just maps the recipe id to the recipe and review. We will make
    it much nicer when we need to use it later on.

    Args:
    recipes: list of recipes
    reviews: list of reviews

    Returns:
    dictionary of recipe id to recipe and reviews
    """
    id_to_recipe = {}

    for recipe in recipes:
        recipe_id = int(recipe["RecipeId"])
        id_to_recipe[recipe_id] = {"recipe": recipe, "reviews": []}

    for review in reviews:
        recipe_id = int(review["RecipeId"])
        if recipe_id in id_to_recipe:
            id_to_recipe[recipe_id]["reviews"].append(review)

    return id_to_recipe
