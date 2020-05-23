#!/usr/bin/env python3
"""
Web app routes using Flask
"""
import json
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from sqlalchemy import create_engine
import plotly
from plotly.graph_objs import Bar
import pickle

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
data = pd.read_sql_table('messages_categories', engine)

# load model
model = pickle.load(open("../models/model.pkl", "rb"))
print (model.get_params())
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Index api route

    Returns:
        rendered template -- rendered template
    """
    
    # extract data needed for visuals
    genre_counts = \
        data.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    
    category_names = data.iloc[:, 5:].columns
    assert len(category_names) == 35
    
    d_count = dict(data[category_names].count())
    d_ones = dict(data[category_names].sum())
    categories_distribution = pd.DataFrame(
        {'1': d_ones,
        '0': {key: d_count[key] - value for key, value in d_ones.items()}})
    categories_distribution.sort_values(by='1', ascending=False, inplace=True)

    # create visuals
    graphs = [
        {
            'data': [
                {
                    "x": genre_names,
                    "y":genre_counts,
                    "type": "bar",
                    "barmode": "group"
                },
            ],
            'layout': {
                'width': 800,
                'height': 400,
                'autosize': False,
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            "data": [
                dict(
                    x=category_names,
                    y=[categories_distribution['0'][category] for category in category_names],
                    type='bar',
                    name='0',
                    # marker=dict(color='rgb(0,255,0)')
                ),
                dict(
                    x=category_names,
                    y=[categories_distribution['1'][category] for category in category_names],
                    type='bar',
                    name='1',
                    # marker=dict(color='rgb(255,0,0)')
                )
            ],
            'layout': {
                'width': 800,
                'height': 600,
                'autosize': False,
                'margin': {
                    'b': 160, 't': 40}, # set e "bottom" margin to 160 px 
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'tickangle': -45
                },
                "barmode": "stack"
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(data.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ Main function
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()