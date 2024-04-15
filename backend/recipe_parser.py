import re


def dietary_restrictions_check(recipe):
    """
    Takes in the recipe and returns a list of booleans correcponding to
    [gluten free, dairy free, vegetarian, vegan, and nut free]
    """
    keywords = re.findall(r'"([^"]*)"', recipe["Keywords"].lower())
    vegan = "vegan" in keywords
    vegetarian = ("vegetarian" in keywords) or vegan
    gluten_free = ("gluten free" in keywords) or ("gluten-free" in keywords)
    dairy_free = ("dairy free" in keywords) or ("dairy-free" in keywords) or vegan
    nut_free = ("nut free" in keywords) or ("nut-free" in keywords)
    return [gluten_free, dairy_free, vegetarian, vegan, nut_free]


def parse_ingredients(recipe):
    """
    Takes in the recipe and returns a list of the ingredients.
    """
    ingredients = recipe["RecipeIngredientParts"]
    ingredients = re.findall(r'"(.*?)"', ingredients)
    return ingredients


def parse_image(recipe):
    """
    Image url is stored in a R format of c(<list of urls>). This function extracts
    the first url from the list.

    Args:
        recipe (dict): The recipe dictionary.

    Returns:
        str: The url of the image.
    """
    image = recipe["Images"]
    image = re.findall(r'"(.*?)"', image)
    return image[0] if image else None


def parse_instructions(recipe):
    """
    Takes in the recipe and returns a string of the instruction steps with a space after each period.
    """
    instructions = recipe["RecipeInstructions"]
    steps = re.findall(r'"(.*?)"', instructions)
    i = ""
    for step in steps:
        i += step + " "
    return i
