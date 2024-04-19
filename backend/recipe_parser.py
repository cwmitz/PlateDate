import re


def tokenize(text):
    """
    Tokenizes the input string into a list of words and removes any punctuation.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list: The list of words from the input string with any punctuation removed.
    """
    return re.findall(r"[a-z]+", text.lower())


def dietary_restrictions_check(recipe):
    """
    Returns a dictionary of dietary restrictions and a boolean value for each restriction.
    We initially assume that the recipe is vegan, vegetarian, gluten free, dairy free, and nut free,
    but if we find key words to the contrary, we will set the corresponding value to False.

    Args:
        recipe (dict): The recipe dictionary.

    Returns:
        dict: A dictionary of dietary restrictions and a boolean value for each restriction.
            in the format of:
            {
            "vegan" : bool,
            "vegetarian" : bool,
            "gluten_free" : bool,
            "dairy_free" : bool,
            "nut_free" : bool,
            }
    """
    # Sets of terms that indicate the recipe is not vegan, vegetarian, gluten free,
    # dairy free, or nut free
    vegetarian_terms = {
        "beef",
        "pork",
        "chicken",
        "fish",
        "seafood",
        "meat",
        "bacon",
        "lamb",
        "turkey",
        "duck",
        "goose",
        "rabbit",
        "venison",
        "veal",
        "ham",
        "salmon",
        "tuna",
        "sardines",
        "anchovies",
        "trout",
        "mackerel",
        "herring",
        "sausage",
        "pepperoni",
        "prosciutto",
        "salami",
        "bologna",
        "pastrami",
        "spam",
        "steak",
    }
    vegan_terms = {
        "milk",
        "cheese",
        "butter",
        "cream",
        "yogurt",
        "gelatin",
        "honey",
        "eggs",
        "egg",
        "lard",
        "casein",
        "whey",
    }
    gluten_free_terms = {
        "wheat",
        "barley",
        "rye",
        "malt",
        "yeast",
        "triticale",
        "bread",
        "pasta",
        "cereal",
        "flour",
        "breadcrumbs",
        "croutons",
        "couscous",
        "farina",
        "semolina",
        "spelt",
        "bulgur",
        "durum",
        "noodle",
        "noodles",
        "spaghetti"
    }
    dairy_free_terms = {
        "milk",
        "cheese",
        "butter",
        "cream",
        "yogurt",
        "lactose",
        "whey",
        "casein",
    }
    nut_free_terms = {
        "almond",
        "hazelnut",
        "walnut",
        "cashew",
        "pecan",
        "pistachio",
        "macadamia",
        "nut",
        "nutty",
    }

    # Initialize all dietary restrictions to True
    vegetarian = True
    vegan = True
    gluten_free = True
    dairy_free = True
    nut_free = True

    # Tokenize the relevant fields of the recipe
    tokens = set(
        tokenize(recipe["Name"])
        + tokenize(recipe["RecipeIngredientParts"][1:])
        + tokenize(recipe["Keywords"][1:])
        + tokenize(recipe["RecipeCategory"])
    )

    # Check if any of the dietary restriction terms are in the tokens
    for term in tokens:
        # If it's not vegetarian, it's not vegan
        if term in vegetarian_terms:
            vegetarian = False
            vegan = False
        if term in dairy_free_terms:
            dairy_free = False
            vegan = False
        if term in vegan_terms:
            vegan = False
        if term in gluten_free_terms:
            gluten_free = False
        if term in nut_free_terms:
            nut_free = False

    return {
        "vegan": vegan,
        "vegetarian": vegetarian,
        "gluten_free": gluten_free,
        "dairy_free": dairy_free,
        "nut_free": nut_free,
    }


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
