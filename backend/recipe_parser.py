import re

def dietary_restrictions_check(recipe):
    """
    TODO: Add docstring.
    """
    raise NotImplementedError


def parse_ingredients(recipe):
    """
    TODO: Add docstring.
    """
    raise NotImplementedError
    # ingredients = recipe["RecipeIngredientParts"]


def parse_instructions(recipe):
    """
    Takes in the recipe and returns a list of the steps in the recipe.
    """
    instructions = recipe["RecipeInstructions"]
    steps = re.findall(r'"(.*?)"', instructions)
    return steps
