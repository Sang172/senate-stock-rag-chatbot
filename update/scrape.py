""" Scrape the stock transactions from Senator periodic filings. """

from bs4 import BeautifulSoup
import numpy as np
import os
import logging
import pandas as pd
import pickle
import requests
import time
from typing import Any, List, Optional
from dotenv import load_dotenv
from google.cloud import storage
from process import process, get_embeddings, load_from_gcs
import json
from google.cloud import aiplatform
import tempfile

load_dotenv()
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
REGION = os.environ.get('GCP_REGION')
INDEX_ID = os.environ.get('INDEX_ID')
INDEX_ENDPOINT_ID = os.environ.get('INDEX_ENDPOINT_ID')
DEPLOYED_INDEX_ID = os.environ.get('DEPLOYED_INDEX_ID')

ROOT = 'https://efdsearch.senate.gov'
LANDING_PAGE_URL = '{}/search/home/'.format(ROOT)
SEARCH_PAGE_URL = '{}/search/'.format(ROOT)
REPORTS_URL = '{}/search/report/data/'.format(ROOT)

BATCH_SIZE = 100
RATE_LIMIT_SECS = 2

PDF_PREFIX = '/search/view/paper/'
LANDING_PAGE_FAIL = 'Failed to fetch filings landing page'

REPORT_COL_NAMES = [
    'transaction_date',
    'file_date',
    'last_name',
    'first_name',
    'order_type',
    'stock_ticker',
    'asset_name',
    'transaction_amount'
]

LOGGER = logging.getLogger(__name__)


def apply_rate_limit(func):
    def with_rate_limit(*args, **kwargs):
        time.sleep(RATE_LIMIT_SECS)
        return func(*args, **kwargs)
    return with_rate_limit


def fetch_csrf_token(session: requests.Session) -> str:
    landing_page_response = session.get(LANDING_PAGE_URL)
    assert landing_page_response.url == LANDING_PAGE_URL, LANDING_PAGE_FAIL

    landing_page_soup = BeautifulSoup(landing_page_response.text, 'lxml')
    csrf_token_element = landing_page_soup.find(attrs={'name': 'csrfmiddlewaretoken'})
    csrf_token_value = csrf_token_element['value'] if csrf_token_element else None

    form_payload = {
        'csrfmiddlewaretoken': csrf_token_value,
        'prohibition_agreement': '1'
    }
    session.post(LANDING_PAGE_URL,
                data=form_payload,
                headers={'Referer': LANDING_PAGE_URL})

    if 'csrftoken' in session.cookies:
        csrf_token = session.cookies['csrftoken']
    else:
        csrf_token = session.cookies['csrf']
    return csrf_token


def get_senator_financial_reports(session: requests.Session) -> List[List[str]]:
    csrf_token = fetch_csrf_token(session)
    offset = 0
    current_batch_reports = query_reports_api(session, offset, csrf_token)
    all_reports: List[List[str]] = []

    while current_batch_reports:
        all_reports.extend(current_batch_reports)
        offset += BATCH_SIZE
        current_batch_reports = query_reports_api(session, offset, csrf_token)

    return all_reports


def query_reports_api(
    session: requests.Session,
    offset: int,
    csrf_token: str
) -> List[List[str]]:

    request_payload = {
        'start': str(offset),
        'length': str(BATCH_SIZE),
        'report_types': '[11]',
        'filer_types': '[]',
        'submitted_start_date': '01/01/2012 00:00:00',
        'submitted_end_date': '',
        'candidate_state': '',
        'senator_state': '',
        'office_id': '',
        'first_name': '',
        'last_name': '',
        'csrfmiddlewaretoken': csrf_token
    }
    LOGGER.info(f'Fetching report rows starting from offset: {offset}')
    response = session.post(REPORTS_URL,
                           data=request_payload,
                           headers={'Referer': SEARCH_PAGE_URL})
    return response.json()['data']


def extract_table_body_from_report_link(session: requests.Session, report_link: str) -> Optional[Any]:

    report_url = f'{ROOT}{report_link}' 
    report_response = session.get(report_url)

    if report_response.url == LANDING_PAGE_URL:
        LOGGER.info('Session expired, resetting CSRF token and session cookie.')
        fetch_csrf_token(session) 
        report_response = session.get(report_url) 

    report_soup = BeautifulSoup(report_response.text, 'lxml')
    table_bodies = report_soup.find_all('tbody')
    if not table_bodies:
        return None
    return table_bodies[0]


def create_transactions_dataframe(session: requests.Session, report_row: List[str]) -> pd.DataFrame:

    first_name, last_name, _, link_html, received_date = report_row
    link_element_soup = BeautifulSoup(link_html, 'lxml')
    link = link_element_soup.a.get('href')

    if link and link.startswith(PDF_PREFIX): 
        return pd.DataFrame()

    transaction_table_body = extract_table_body_from_report_link(session, link)
    if transaction_table_body is None:
        return pd.DataFrame()

    stock_transactions = []
    for transaction_row in transaction_table_body.find_all('tr'):
        transaction_columns = [col.get_text() for col in transaction_row.find_all('td')]
        (transaction_date, _, _, stock_ticker, asset_name, asset_type, order_type, transaction_amount) = transaction_columns[1], transaction_columns[2], transaction_columns[3], transaction_columns[4], transaction_columns[5], transaction_columns[6], transaction_columns[7], transaction_columns[8] # directly unpack relevant columns

        if asset_type != 'Stock' and stock_ticker.strip() in ('--', ''): # more descriptive variable name stock_ticker
            continue

        stock_transactions.append([
            transaction_date,
            received_date,
            last_name,
            first_name,
            order_type,
            stock_ticker,
            asset_name,
            transaction_amount
        ])

    return pd.DataFrame(stock_transactions, columns=REPORT_COL_NAMES)


def main() -> pd.DataFrame:
    """
    Main function to orchestrate fetching and processing of senator financial reports.

    Initializes a session, applies rate limiting, retrieves reports,
    processes each report to extract transactions, and compiles all transactions
    into a single Pandas DataFrame.
    """
    LOGGER.info('Initializing HTTP session with rate limiting.') 
    session = requests.Session()
    session.get = apply_rate_limit(session.get) 
    session.post = apply_rate_limit(session.post) 

    all_reports = get_senator_financial_reports(session)
    all_transactions = pd.DataFrame()

    for index, report_row in enumerate(all_reports): 
        if index % 10 == 0:
            LOGGER.info(f'Processing report #{index}/{len(all_reports)}') 
            LOGGER.info(f'Aggregated transactions so far: {len(all_transactions)}') 
        transactions = create_transactions_dataframe(session, report_row)
        all_transactions = pd.concat([all_transactions, transactions], ignore_index=True)

    LOGGER.info(f'Total transactions fetched: {len(all_transactions)}') 
    return all_transactions


def save_to_gcs(bucket_name, filename, dataframe):
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    temp_file_path = f"temp_{filename}"
    with open(temp_file_path, "wb") as temp_file:
        pickle.dump(dataframe, temp_file)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)


def upload_embeddings_to_gcs(embeddings, doc_ids, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    embeddings_data = []
    for doc_id, embedding in zip(doc_ids, embeddings):
        embeddings_data.append(
            {"id": doc_id, "embedding": embedding.tolist()}
        )

    data_str = "\n".join([json.dumps(item) for item in embeddings_data])

    blob.upload_from_string(data_str)
    LOGGER.info(f"Embeddings uploaded to gs://{bucket_name}/{destination_blob_name}")



def process_data(bucket_name, filename):
    LOGGER.info("Reading data from senate_trade.pickle")
    data = load_from_gcs(bucket_name, filename)
    LOGGER.info("Successfully read data")

    LOGGER.info("Processing data")
    documents = process(data)
    LOGGER.info("Data processing complete")

    doc_embeddings = []
    LOGGER.info(f"Start creating vector embeddings for {len(documents)} documents")
    doc_embeddings = get_embeddings(documents)
    LOGGER.info("Embedding creation complete")

    doc_embeddings = np.array(doc_embeddings)

    LOGGER.info("Saving documents to documents.pickle in GCS bucket")
    save_to_gcs(bucket_name, 'documents.pickle', documents)
    LOGGER.info("Successfully saved documents to GCS bucket")


    LOGGER.info("Saving document embeddings to doc_embeddings.pickle and update/embeddings.json in GCS bucket")
    save_to_gcs(bucket_name, 'doc_embeddings.pickle', doc_embeddings)
    doc_ids = [str(i) for i in range(len(doc_embeddings))]
    upload_embeddings_to_gcs(doc_embeddings, doc_ids, GCS_BUCKET_NAME, 'update/embeddings.json')
    LOGGER.info("Successfully saved document embeddings to GCS bucket")

    LOGGER.info("Start updating Vertex AI vector search index.")
    aiplatform.init(project=PROJECT_ID, location=REGION)
    index = aiplatform.MatchingEngineIndex(INDEX_ID)
    operation = index.update_embeddings(contents_delta_uri='gs://senate-stock-rag-chatbot/update/', is_complete_overwrite=True)
    LOGGER.info("Vertex AI vector search index update complete.")

if __name__ == '__main__':
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    senator_txs = main()
    LOGGER.info('Dumping to .pickle')

    
    save_to_gcs(GCS_BUCKET_NAME, 'senate_trade.pickle', senator_txs)
    LOGGER.info('Successfully uploaded senate_trade.pickle to GCS bucket: {}'.format(GCS_BUCKET_NAME))

    process_data(GCS_BUCKET_NAME, 'senate_trade.pickle')
