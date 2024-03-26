import pandas as pd
import re

def url_builder(recipe_id, name):
    name = re.sub(' ', '-', name)
    name = name.lower()
    url = f'https://www.food.com/recipe/{name}-{recipe_id}'
    return url

def build_recipe_url_data(url_builder, recipe_data, out_file):
    recipe_data['Url'] = recipe_data.apply(lambda x: url_builder(x['RecipeId'],x['Name']), axis=1)
    recipes_urls = recipe_data[['RecipeId', 'Name', 'Url']]
    recipes_urls.to_csv(out_file, index = False)






