import json
import os
import algorithm

import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from pathlib import Path


# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, "data/id_to_recipe.json")

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, "r") as file:
    data = json.load(file)
recipes_list = list(data.values())
df = pd.json_normalize(recipes_list)


app = Flask(__name__)
CORS(app)


idf_path = os.path.join(current_directory, 'data/idf.json')
inv_idx_path = os.path.join(current_directory, 'data/inv_idx.json')
recipe_norms_path = os.path.join(
    current_directory, 'data/recipe_norms.json')
id_to_recipe_path = os.path.join(
    current_directory, 'data/id_to_recipe.json')

with open(idf_path, 'r') as f:
    idf = json.load(f)

with open(inv_idx_path, 'r') as f:
    inverted_index = json.load(f)

with open(recipe_norms_path, 'r') as f:
    recipe_norms = {int(k): v for k, v in json.load(f).items()}

with open(id_to_recipe_path, 'r') as f:
    id_to_recipe = json.load(f)


def cosine_search(queries):
    top_10_recipes = algorithm.algorithm(
        queries, inverted_index, idf, recipe_norms)

    top_10_ids = [recipe_id for recipe_id, _ in top_10_recipes]

    details = [
        {"name": id_to_recipe[str(recipe_id)]["name"],
         "instructions": id_to_recipe[str(recipe_id)]["instructions"],
         "aggregated_rating": id_to_recipe[str(recipe_id)]["aggregated_rating"]
         }
        for recipe_id in top_10_ids if str(recipe_id) in id_to_recipe
    ]

    return json.dumps(details, indent=4)


queries = ["lemon dessert"]
s = cosine_search(queries)
# print(s)


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/recipes")
def episodes_search():
    text = request.args.get("title")
    return cosine_search(text)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
