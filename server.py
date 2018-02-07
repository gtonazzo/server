#!flask/bin/python
from flask import Flask, jsonify, request, abort
from sklearn.externals import joblib
import pandas as pd
import json
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load Articles
metadata = pd.read_csv('data/ArticlesWithAbstracts.csv', sep=';', low_memory=False)

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1, 2))

#Replace NaN with an empty string
metadata['abstract'] = metadata['abstract'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['abstract'])

@app.route('/todo/api/v1.0/articles', methods=['POST'])
def get_similar_articles():
    if not request.json or not 'search' in request.json:
        abort(400)

    text = request.json['search']
    invec = tfidf.transform([text]).toarray()
    cosine_sim = linear_kernel(tfidf_matrix, invec)

    sim_scores = list(enumerate(cosine_sim))
    # Sort the atricles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar articles
    sim_scores = sim_scores[1:11]
   
    # Get the article indices
    result = ' { "articles" :  ['
    article_indices = [i[0] for i in sim_scores]
    for i in sim_scores:
        result = result + '{' 
        result = result +   ' "art_index" : '  + '"' + str(i[0]) + '", '
        result = result +   ' "similarity" : '  + '"' +str(i[1][0]) + '" '
        result = result +  '},'
        #print(str(i[0]) + ' ' + str(i[1][0]) + ' ' + str(metadata['art_id'][i[0]]) + ' ' + str(metadata['title'][i[0]]) )        
    
    result = result[:-1]
    result = result + ']}'

    return result


if __name__ == '__main__':
    app.run(debug=True)
