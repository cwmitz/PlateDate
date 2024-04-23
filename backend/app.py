import json
import os

import algorithm
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from fuzzywuzzy import fuzz, process
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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

id_to_recipe_path = os.path.join(current_directory, "data/id_to_recipe.json")
inverted_index_path = os.path.join(current_directory, "data/inv_idx.json")


with open(id_to_recipe_path, "r") as f:
    id_to_recipe = json.load(f)

with open(inverted_index_path, "r") as f:
    inverted_index = json.load(f)

corpus = []
for _, recipe in id_to_recipe.items():
    text = (
        " ".join(recipe["ingredients"])
        + " "
        + recipe["description"]
        + " "
        + recipe["instructions"]
    )
    corpus.append(text)


vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(corpus)

n_components = 150
svd_model = TruncatedSVD(n_components=n_components)
svd = svd_model.fit_transform(tfidf_matrix)


def cosine_search(queries, dietary_restrictions):
    # Preprocess each query to find the closest match in the corpus
    preprocessed_queries = []
    for query in queries:
        tokens = query.split()
        corrected_tokens = []
        for token in tokens:
            if token not in inverted_index:
                corrected_token = process.extractOne(
                    token, inverted_index.keys(), scorer=fuzz.ratio
                )
                if corrected_token and corrected_token[1] >= 80:
                    corrected_tokens.append(corrected_token[0])
            else:
                corrected_tokens.append(token)
        preprocessed_queries.append(" ".join(corrected_tokens))

    top_10_recipes, sim_scores = algorithm.algorithm(
        preprocessed_queries,
        dietary_restrictions,
        id_to_recipe,
        vectorizer,
        svd_model,
        svd,
    )

    top_10_ids = [recipe_id for recipe_id, _ in top_10_recipes]

    details = [
        {
            "name": id_to_recipe[str(recipe_id)]["name"],
            "instructions": id_to_recipe[str(recipe_id)]["instructions"],
            "aggregated_rating": id_to_recipe[str(recipe_id)]["aggregated_rating"],
            "image": id_to_recipe[str(recipe_id)]["image"],
            "Url": id_to_recipe[str(recipe_id)]["Url"],
            "total_time": id_to_recipe[str(recipe_id)].get("total_time", "PT0M"),
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
        "gluten_free": request.args.get("gluten_free") == "true",
        "dairy_free": request.args.get("dairy_free") == "true",
        "nut_free": request.args.get("nut_free") == "true",
    }
    search_results = cosine_search(texts, dietary_restrictions)
    return jsonify(search_results)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
