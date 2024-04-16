import json
import os

import algorithm
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

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


idf_path = os.path.join(current_directory, "data/idf.json")
inv_idx_path = os.path.join(current_directory, "data/inv_idx.json")
recipe_norms_path = os.path.join(current_directory, "data/recipe_norms.json")
id_to_recipe_path = os.path.join(current_directory, "data/id_to_recipe.json")

with open(idf_path, "r") as f:
    idf = json.load(f)

with open(inv_idx_path, "r") as f:
    inverted_index = json.load(f)

with open(recipe_norms_path, "r") as f:
    recipe_norms = {int(k): v for k, v in json.load(f).items()}

with open(id_to_recipe_path, "r") as f:
    id_to_recipe = json.load(f)


def cosine_search(queries, dietary_restrictions, time_limit):
    top_10_recipes, sim_scores = algorithm.algorithm(
        queries,
        dietary_restrictions,
        inverted_index,
        idf,
        recipe_norms,
        id_to_recipe,
        time_limit
    )

    top_10_ids = [recipe_id for recipe_id, _ in top_10_recipes]

    details = [
        {
            "name": id_to_recipe[str(recipe_id)]["name"],
            "instructions": id_to_recipe[str(recipe_id)]["instructions"],
            "aggregated_rating": id_to_recipe[str(recipe_id)]["aggregated_rating"],
            "image": id_to_recipe[str(recipe_id)]["image"],
            "Url": id_to_recipe[str(recipe_id)]["Url"],
            "similarity_scores": sim_scores[recipe_id],
        }
        for recipe_id in top_10_ids
        if str(recipe_id) in id_to_recipe
    ]

    return details


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/recipes")
def recipes_search():
    texts = [
        request.args.get(f"title{i}")
        for i in range(len(request.args))
        if f"title{i}" in request.args
    ]
    dietary_restrictions = {
        "vegetarian": request.args.get("vegetarian") == "true",
        "vegan": request.args.get("vegan") == "true",
        "gluten-free": request.args.get("gluten-free") == "true",
        "dairy-free": request.args.get("dairy-free") == "true",
    }
    time_limit = request.args.get("timeLimit")  
    search_results = cosine_search(texts, dietary_restrictions, time_limit)
    return jsonify(search_results)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
