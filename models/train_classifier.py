import sys
import nltk
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Messages", engine)

    # Split out targets and drop any category columns which sum to zero
    Y = df.iloc[:,4:]
    Y = Y.loc[:, (Y!=0).any(0)]
    # Extract list of target categories
    categories = Y.columns.tolist()
    # Convert to array
    Y = Y.values

    # Split out 'message' column and convert to 1d array
    X = np.ravel(df[['message']].values)
    
    return X, Y, categories


def tokenize(text):
    
    # Remove punctuation, covert to lowercase and strip whitespace
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Create tokens
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = stopwords.words("english")
    tokens = [w for w in tokens if w not in stop_words]
    
    # Create lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize, strip, and covert to lower case
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens  


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier()))
    ])
    
    parameters = {
        'clf__estimator__max_iter': [10, 30],
        'clf__estimator__penalty': ['l2', 'elasticnet'],
        'vect__max_df': (0.5, 0.75),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0, Y_test.shape[1]):
        precision, recall, fscore, support = precision_recall_fscore_support(Y_test[:,i], Y_pred[:,i])
        print('Category:' , category_names[i].upper(), '\n\t', 
              'Precision: ', precision[1], '   ',
              'Recall: ', recall[1],  '   ',
              'F1-Score: ', fscore[1],
              '\n')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()