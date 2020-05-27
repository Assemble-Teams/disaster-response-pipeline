#!/usr/bin/env python3
"""
Model training script
"""
import sys
import logging
import pickle
import time
import re
import matplotlib.pyplot as plt
import json
import os

import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def load_data(database_filepath):
    """
    Load data from sqlite database file

    Arguments:
        database_filepath {string} -- database file path

    Returns:
        Array[string], Array{int},Array{string}  -- feature, target variables and category names
    """
    engine = create_engine('sqlite:///../data/{}'.format(database_filepath))
    dataframe = pd.read_sql_table('messages_categories', engine)
    feature_values = dataframe.message.values
    target_variables = dataframe.iloc[:, 5:]
    return feature_values, target_variables, target_variables.keys()

def tokenize(text):
    """Text data Tokenization and stop words removal

    Arguments:
        text {string} -- text string
    Returns:
        Array{string}: Array of tokens
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word, pos='v').strip()
        for word in words
        if word not in stopwords.words("english") and len(word) > 1
    ]

def build_model():
    """Build model
    Returns:
        GridSearchCV object -- Classifier object
    """
    pipeline = Pipeline([
      ('features', FeatureUnion([
        ('text', Pipeline([
            ('vect', CountVectorizer(
                tokenizer=tokenize,
                ngram_range=(1, 2),
                max_features=None)
            ),
            ('reduce_dim', TruncatedSVD()),
            ('tfidf', TfidfTransformer())
        ]))
        ], n_jobs=1)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            min_samples_split=10, criterion='entropy', n_estimators=120,
            max_features="auto")))
    ])
    print("Pipeline Parameters: {}".format(pipeline.get_params()))
    parameters = {
        'features__text__vect__max_df': (0.9, 1.0),
        'features__text__vect__max_features': (None, 1000, 5000),
        # 'features__text__vect__ngram_range': [(1, 1), (1, 2)], # unigrams or bigrams
        # 'features__text__reduce_dim__n_components': [1, 2, 10],
        # 'clf__estimator__max_depth': [None, 10, 20],
        'clf__estimator__min_samples_leaf': [4, 10],
    #     'clf__estimator__n_estimators': [1, 10, 50],
        # 'clf__estimator__criterion': ['gini', 'entropy'],
    #     'clf__estimator__max_features': ['sqrt', 'log2'],
    }
    clf = GridSearchCV(estimator=pipeline, param_grid=parameters,
        n_jobs=-1, verbose=2)
    return clf

def evaluate_model(model, x_test, y_test, category_names):
    """Evaluate the MultiOutput Classifier model
    Arguments:
        model {Model object} -- MultiOutput classifier model object
        x_test {ndarray} -- Test examples array
        y_test {datarframe} -- DataFrame of test data
        category_names {list of strings} -- Category names list
    """
    tic = time.perf_counter()
    y_pred_test = model.predict(x_test)
    toc = time.perf_counter()
    nb_examples = len(x_test)
    print(f"Predicting {nb_examples} examples in {toc - tic:0.4f} seconds")

    accuracy = (y_pred_test == y_test).mean()

    print("Accuracy by category:", accuracy)
    print("Average accuracy", accuracy.mean())
    print(classification_report(y_test, y_pred_test))

    # print("\nBest Parameters:", model.best_params_)
    f1_scores = {}
    for index, column in enumerate(y_test.columns):
        report = classification_report(
            y_test[column], y_pred_test[:, index],
            digits=4, output_dict=True)
        f1_scores[column] = report["macro avg"]["f1-score"]
    f1_scores = dict(sorted(f1_scores.items(), key=lambda kv: kv[1], reverse=True))
    fig_data = pd.DataFrame({"categories": list(f1_scores.keys()), "F1 Score": list(f1_scores.values())})
    fig_data.plot.bar(figsize=(16, 6), x="categories", y="F1 Score")
    plt.tight_layout()
    plt.savefig("f1_scores_mlo_classifier.png")

def save_model(model, model_filepath):
    """Serialize model object with pickle module
    Arguments:
        model {model object} -- Model object
        model_filepath {string} -- Model output file path
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    # dump model parameters to a json file
    # params_filename = "{}_params.json".format(os.path.splitext(model_filepath)[0])
    # with open(params_filename, "w") as outfile:
    #     json.dump(model.get_params(), outfile)

def main():
    """Main function
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x_data, labels, category_names = load_data(database_filepath)
        print(category_names)
        x_train, x_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.2, random_state=42)
        print('Building model...')
        model = build_model()

        print('Training model...')
        tic = time.perf_counter()
        model.fit(x_train, y_train)
        toc = time.perf_counter()

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, "../models/{}".format(model_filepath))

        print('Trained model saved!')
        print(f"Model trained in {toc - tic:0.4f} seconds")

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
