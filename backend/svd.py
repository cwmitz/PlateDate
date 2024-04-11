import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import pandas as pd
from sklearn.preprocessing import normalize


def create_svd_docs_similarity(recipes):
    vectorizer = TfidfVectorizer(max_df = .7)
    documents = []
    for recipe in recipes:
        recipe['Text'] = str(recipe['Name']) + ' ' + str(recipe['RecipeIngredientParts']) + ' ' + str(recipe['Keywords']) + ' ' + str(recipe['RecipeCategory'])
        documents.append({'RecipeId': recipe['RecipeId'], 'Text' : recipe['Text']})
    td_matrix = vectorizer.fit_transform([x['Text'] for x in documents])
    # play around to find best # of dimensions, just using 40 for now
    words_compressed, _, docs_compressed = svds(td_matrix, k=40)
    docs_compressed
    docs_compressed = docs_compressed.transpose()
    word_to_index = vectorizer.vocabulary_
    index_to_word = {i:t for t,i in word_to_index.items()}
    words_compressed_normed = normalize(words_compressed, axis = 1)
    docs_compressed_normed = normalize(docs_compressed)
    return documents, word_to_index, words_compressed_normed, docs_compressed_normed

def closest_recipes_to_word(documents, word_to_index, words_compressed_normed, docs_compressed_normed, word_in, k = 5):
    if word_in not in word_to_index: return "Not in vocab."
    sims = docs_compressed_normed.dot(words_compressed_normed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(i, documents[i]['Text'],sims[i]) for i in asort[1:]]

def print_closest_recipes(results):
    for recipe in results:
        print(recipe)
    




