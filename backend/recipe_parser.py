import re

def dietary_restrictions_check(recipe):
    """
    TODO: Add docstring.
    """
    raise NotImplementedError


def parse_ingredients(recipe):
    """
    Takes in the recipe and returns a list of the ingredients.
    """
    ingredients = recipe["RecipeIngredientParts"]
    ingredients = re.findall(r'"(.*?)"', ingredients)
    return ingredients

def parse_instructions(recipe):
    """
    Takes in the recipe and returns a list of the instruction steps.
    """
    instructions = recipe["RecipeInstructions"]
    steps = re.findall(r'"(.*?)"', instructions)
    return steps