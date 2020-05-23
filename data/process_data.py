#!/usr/bin/env python3
"""
Data processing script
"""
import sys
import os

from sqlalchemy import create_engine
import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet'])


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories DataSets

    Arguments:
        messages_filepath {string} -- messages csv file path
        categories_filepath {string} -- categories csv file

    Returns:
        DataFrame -- DataSet DataFrame object
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    data = messages.merge(categories, how='outer', on=['id'])
    return data

def clean_data(data):
    """Clean data:
        - Merge messages and categories DataSet
        - Split categories into separate category columns
        - Convert category values to just numbers 0 or 1
        - Remove duplicates
    Arguments:
        data {DataFrame} -- DataFrame object

    Returns:
        DataFrame -- Cleaned Dataset DataFrame
    """
    category_colnames = [column[:-2] for column in data.iloc[0]["categories"].split(";")]
    data[category_colnames] = data.categories.str.split(";",expand=True)
    data.drop(["categories"], axis=1, inplace=True)

    for column in category_colnames:
      data[column] = data[column].apply(lambda x: int(x[-1:]))
    data.drop_duplicates(inplace=True)
    return data

def save_data(data, database_filename):
    """Serialize the model in pickle format

    Arguments:
        data {DataFrame} -- DataFrame object
        database_filename {string} -- Database filename
    """
    os.remove(database_filename)
    engine = create_engine('sqlite:///{}'.format(database_filename))
    data.to_sql('messages_categories', engine, index=False)


def main():
    """Main function
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        data = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        data = clean_data(data)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(data, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
