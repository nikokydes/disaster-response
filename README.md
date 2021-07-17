# Disaster Response Pipeline Project

## Project Description
This is a Udacity project for their Data Scientist Nanodegree program. The purpose of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. This application uses a machine learning pipeline to categorize these events so that the messages may be sent to an appropriate disaster relief agency.

This project will includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

## Installation
This application was created using a standard Anaconda Python installation with a core python version of 3.6.3.  Libraries used include:
- _Scikit-Learn_ for machine learning
- _Flask_ and _Ployly_ for creating the web application and visuals
- _NLTK_ for natural language processing
- _Pickle_ for serializing the trained machine learning model
- _SQLAlchemy_ for database interactions

## File Descriptions
The file structure of the repository is as follows:

- The **/app/run.py** file loads the data, sets up the visuals and launches a Flask web app
- The **/app/templates/master.html** file is the main page of the web app
- The **/app/templates/go.html** file creates the classification results of web app
- The **/data/*.csv** files contain the text message data used to train the NLP model
- The **/data/process_data.py** file is a python script which processes the text message data used to train the NLP model
- The **/data/DisasterResponse.db** file is a SQLite database containing the processed text message data
- The **/models/train_classifier.py** which builds and trains the NLP model on the message data and pickles the trained model
- The **/models/classifier.pkl** file is the trained NLP model which may be used to classify new messages

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
This project is part of the [Udacity](https://www.udacity.com) Data Science Nanodegree program and is motivated by data from [Figure Eight's](https://www.figure-eight.com) disaster response efforts.

## Authors
- [Nicolaos Kydes](https://github.com/nikokydes)

## License
[MIT](https://opensource.org/licenses/MIT)

