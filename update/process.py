import pandas as pd
import json
import re
import os
import pickle
from datetime import datetime
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from google.cloud import storage

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')





def get_embeddings(texts, model="models/text-embedding-004", batch_size=50):
    """Generates embeddings for a list of texts using a specified embedding model.
    Args:
        texts: A list of strings to generate embeddings for.
        model: The path to the embedding model (default: "models/text-embedding-004").
        batch_size: The number of texts to process in a single batch (default: 50).
    Returns:
        A list of embeddings, where each embedding is a list of floats representing
        the vector embedding of the corresponding text.
    """
    all_embeddings = []
    num_texts = len(texts)

    for i in range(0, num_texts, batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Getting embeddings for documents {i}-{i + batch_size}")

        result = genai.embed_content(
            model=model,
            content=batch
        )
        embeddings = result['embedding']
        all_embeddings.extend(embeddings)

    return all_embeddings





def dataframe_to_string(df):
    """Converts a Pandas DataFrame to a formatted string representation.
    Args:
        df: The Pandas DataFrame to convert.
    Returns:
        A string representation of the DataFrame.
    """
    string_representation = ""
    for i, row in df.iterrows():
        string_representation += f"Transaction {i + 1}:\n"
        for col, value in row.items():
            if col not in ['Ticker', 'Month', 'Name']:
                if isinstance(value, pd.Timestamp):
                    value = value.strftime('%Y-%m-%d')
                string_representation += f"{col}: {value}\n"
        string_representation += "\n"
    return string_representation



def reduce_whitespace(text):
    """Reduces sequences of three or more whitespace characters to a comma and a space.
    Args:
        text: The input string.
    Returns:
        The string with reduced whitespace.
    """
    return re.sub(r"\s{3,}", ", ", text)


def process(data):

    #data cleaning
    for c in data.columns:
        data[c] = data[c].apply(lambda x: x.strip())
    for c in data.columns:
        if c=='tx_date':
            data[c] = pd.to_datetime(data[c])

    #data cleaning
    data['asset_name'] = data['asset_name'].apply(reduce_whitespace)
    data['Name'] = data['first_name'] + ' ' + data['last_name']
    data = data[data['order_type']!='Exchange']
    data.loc[data['ticker'] == 'SPY160219P00180000', 'ticker'] = 'SPY'
    data = data.drop(columns=['file_date','first_name','last_name'])
    data['tx_month'] = data['tx_date'].dt.strftime('%B %Y')
    data['Name'] = data['Name'].apply(lambda x: x[:-1] if x[-1]==',' else x)
    data.columns = ['Date','Order Type','Ticker','Asset Name','Amount','Name', 'Month']

    strings = []

    #generate strings for each name, ticker, month combination
    data_grouped = data.groupby(['Name', 'Ticker', 'Month'])
    strings = []
    for g in data_grouped.groups:
        if g[1]!='--':
            name = g[0]
            ticker = g[1]
            month = g[2]
            sub_df = data_grouped.get_group(g).copy()
            sub_df = sub_df.sort_values('Date')
            sub_df = sub_df.reset_index(drop=True)
            title = f"Senator {name}'s Transactions Related to the Ticker {ticker} in {month}\n\n"
            content = dataframe_to_string(sub_df)
            strings.append(title+content)


    #generate string for popular tickers
    summary_ticker = pd.DataFrame(data[['Ticker', 'Asset Name']].groupby('Ticker').count().sort_values('Asset Name', ascending=False))
    title = f'Summary of Transaction Records Based on Ticker\n\n'
    title += 'Contains information on which ticker was most actively traded.\n\n'

    for i in range(summary_ticker.shape[0]):
        ticker = summary_ticker.index[i]
        count = summary_ticker.iloc[i]['Asset Name']
        if count>=20 and ticker!='--':
            content = f'The ticker {ticker} had {count} transactions by senators.\n'
            title += content
    strings.append(title)


    #generate string for active traders
    summary_name = pd.DataFrame(data[['Name','Asset Name']].groupby('Name').count().sort_values('Asset Name', ascending=False))
    title = f'Summary of Transaction Records Based on Senator\n\n'
    title += 'Contains information on which senators were the most active traders.\n\n'
    for i in range(summary_name.shape[0]):
        name = summary_name.index[i]
        count = summary_name.iloc[i]['Asset Name']
        if ticker!='--':
            content = f'Senator {name} made {count} transactions during their term in office.\n'
            title += content
    strings.append(title)


    #generate strings for monthly aggregations
    data['Month'] = pd.to_datetime(data['Month'], format='%B %Y')
    data['Month'] = data['Month'].dt.strftime('%Y-%m')
    summary_month = pd.DataFrame(data[['Month','Asset Name']].groupby('Month').count().sort_index())
    title = f'Summary of Transaction Records Based on Month\n\n'
    title += 'Contains information on monthly senator transaction counts in chronological order.\n\n'

    for i in range(summary_month.shape[0]):
        month = summary_month.index[i]
        count = summary_month.iloc[i]['Asset Name']
        if ticker!='--':
            content = f'{count} senator transactions in {month} were reported and disclosed.\n'
            title += content
    strings.append(title)


    #generate strings for yearly aggregations
    data['Year'] = pd.to_datetime(data['Month']).dt.year
    summary_year = pd.DataFrame(data[['Year','Asset Name']].groupby('Year').count().sort_index())
    title = f'Summary of Transaction Records Based on Year\n\n'
    title += 'Contains information on yearly senator transaction counts in chronological order.\n\n'

    for i in range(summary_year.shape[0]):
        year = summary_year.index[i]
        count = summary_year.iloc[i]['Asset Name']
        if ticker!='--':
            content = f'{count} senator transactions in {year} were reported and disclosed.\n'
            title += content
    strings.append(title)

    return strings



def load_from_gcs(bucket_name, filename):
    """Loads a pickled object from a Google Cloud Storage (GCS) bucket.
    Args:
        bucket_name: The string name of the GCS bucket.
        filename: The string name of the file (blob) within the bucket.
    Returns:
        The unpickled data from the file.  Returns None if any error occurs
        during the process (e.g., file not found, invalid pickle data).
    """
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    with blob.open("rb") as file:
        data = pickle.load(file)
    return data



def save_to_gcs(bucket_name, filename, data):
    """Saves a data to a Google Cloud Storage (GCS) bucket as a pickled object.

    Args:
        bucket_name: The string name of the GCS bucket.
        filename: The string name of the file (blob) to be created in the bucket.
        data: object to save.
    """
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    temp_file_path = f"temp_{filename}"
    with open(temp_file_path, "wb") as temp_file:
        pickle.dump(data, temp_file)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)