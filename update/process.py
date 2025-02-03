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
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

def get_embedding(text, model="models/text-embedding-004"):
    result = genai.embed_content(
        model=model,
        content=text
    )
    return result['embedding']


def dataframe_to_string(df):
  string_representation = ""
  for i, row in df.iterrows():
    string_representation += f"Transaction {i+1}:\n"
    for col, value in row.items():
      if col not in ['Ticker', 'Month', 'Name']:
        if isinstance(value, pd.Timestamp):
          value = value.strftime('%Y-%m-%d')
        string_representation += f"{col}: {value}\n"
    string_representation += "\n"
  return string_representation

def reduce_whitespace(text):
    return re.sub(r"\s{3,}", ", ", text)

def process(data):

    for c in data.columns:
        data[c] = data[c].apply(lambda x: x.strip())
    for c in data.columns:
        if c=='tx_date':
            data[c] = pd.to_datetime(data[c])

    data['asset_name'] = data['asset_name'].apply(reduce_whitespace)
    data['Name'] = data['first_name'] + ' ' + data['last_name']
    data = data[data['order_type']!='Exchange']
    data.loc[data['ticker'] == 'SPY160219P00180000', 'ticker'] = 'SPY'
    data = data.drop(columns=['file_date','first_name','last_name'])
    data['tx_year'] = data['tx_date'].dt.strftime('%B %Y')
    data['Name'] = data['Name'].apply(lambda x: x[:-1] if x[-1]==',' else x)
    data.columns = ['Date','Order Type','Ticker','Asset Name','Amount','Name', 'Month']

    strings = []

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
            # assets = list(sub_df['Asset Name'].unique())
            # title += f'Involves assets such as {', '.join(assets)}.\n\n'
            content = dataframe_to_string(sub_df)
            strings.append(title+content)


    summary_ticker = pd.DataFrame(data[['Ticker', 'Asset Name']].groupby('Ticker').count().sort_values('Asset Name', ascending=False))
    title = f'Summary of Transaction Records Based on Ticker\n\n'
    title += 'Contain information on which ticker was most actively traded.\n\n'

    for i in range(summary_ticker.shape[0]):
        ticker = summary_ticker.index[i]
        count = summary_ticker.iloc[i]['Asset Name']
        if count>=20 and ticker!='--':
            content = f'The ticker {ticker} had {count} transactions by senators.\n'
            title += content
    strings.append(title)

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
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    with blob.open("rb") as file:
        data = pickle.load(file)
    return data

def save_to_gcs(bucket_name, filename, dataframe):
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    temp_file_path = f"temp_{filename}"
    with open(temp_file_path, "wb") as temp_file:
        pickle.dump(dataframe, temp_file)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)

"""
if __name__=='__main__':

    logger.info("Reading data from senate_trade.pickle")
    data = load_from_gcs(GCS_BUCKET_NAME, 'senate_trade.pickle')
    logger.info("Successfully read data from senate_trade.pickle")

    logger.info("Processing data")
    documents = process(data)
    logger.info("Data processing complete")

    doc_embeddings = []
    logger.info(f"Start creating vector embeddings for {len(documents)} documents")
    for i, doc in enumerate(documents):
        if (i+1)%100==0:
            logger.info(f"Embedding created for {i+1} th document")
        doc_embeddings.append(get_embedding(doc))
    logger.info("Embedding creation complete")

    doc_embeddings = np.array(doc_embeddings)

    logger.info("Saving documents to documents.pickle in GCS bucket")
    save_to_gcs(GCS_BUCKET_NAME, 'documents.pickle', documents)
    logger.info("Successfully saved documents to documents.pickle in GCS bucket")

    logger.info("Saving document embeddings to doc_embeddings.pickle in GCS bucket")
    save_to_gcs(GCS_BUCKET_NAME, 'doc_embeddings.pickle', doc_embeddings)
    logger.info("Successfully saved document embeddings to doc_embeddings.pickle in GCS bucket")


    logger.info("Data processed and saved as pickle files in GCS bucket.")
"""