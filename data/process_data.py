"""
Python script to read in message and category data from CSV files, clean and merge them,
and store in an SQLite database table.

Project: Udacity Disaster Response Pipeline Project (Project 2)

Inputs:
    1. File name and path to the CSV file containing the message data
    2. File name and path to the CSV file containig the message category data
    3. File name and path of the output SQLite database file (*.db)
    
Outputs:
    1. An SQLite database containing a 'Messages' table containing 
        disaster text messages and category data
    
Example call:
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
"""

# Import statements
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to read in the messages and category CSV files into Pandas dataframes
    and them into a single dataframe
    
    Inputs:
        messages_filepath: File path to the messages CSV file
        categories_filepath: File path to the categories CSV file
    Output:
        df: Merged dataframe containing both message and category data
    """   
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Function to convert the raw combined 'categories' column into a set of 
    binary indicator columns for each category.
    
    Input:
        df: Merged dataframe containing both message and category data
    Output:
        df: Dataframe with category data split out into individual columns
    """ 

    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # Select the first row of the categories dataframe and use 
    # this row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0]).values

    # Rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: min(1,pd.to_numeric(x.split('-')[1])))

    # Replace `categories` column in `df` with new category columns.
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates.
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Function to save the cleaned and prepared data to an SQLite database
    
    Input:
        df: Dataframe with category data split out into individual columns
        database_filename: File name and path to output SQLite database
    Output:
        None
    """ 
    
    # Save the cleaned dataframe to a SQLite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
    """
    Main function to execute the data preparation and output steps.
    """ 
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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