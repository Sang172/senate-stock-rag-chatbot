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
from process import process, get_embeddings, load_from_gcs, save_to_gcs
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
    'tx_date',
    'file_date',
    'last_name',
    'first_name',
    'order_type',
    'ticker',
    'asset_name',
    'tx_amount'
]

LOGGER = logging.getLogger(__name__)



def add_rate_limit(f):
    """Adds a rate limit to a function by pausing execution for a predefined time.
    Args:
        f: The function to be rate-limited.
    Returns:
        The rate-limited function.
    """
    def with_rate_limit(*args, **kw):
        time.sleep(RATE_LIMIT_SECS)
        return f(*args, **kw)
    return with_rate_limit


def _csrf(client: requests.Session) -> str:
    """Retrieves the CSRF token from a web page.
    This function simulates the initial interaction with a website to obtain
    a CSRF token. It first loads the landing page, extracts the token from
    a form, submits the form, and then retrieves the token from the client's
    cookies.
    Args:
        client: A requests.Session object to manage cookies and connections.
    Returns:
        The CSRF token as a string.
    """
    landing_page_response = client.get(LANDING_PAGE_URL)
    assert landing_page_response.url == LANDING_PAGE_URL, LANDING_PAGE_FAIL

    landing_page = BeautifulSoup(landing_page_response.text, 'lxml')
    form_csrf = landing_page.find(
        attrs={'name': 'csrfmiddlewaretoken'}
    )['value']
    form_payload = {
        'csrfmiddlewaretoken': form_csrf,
        'prohibition_agreement': '1'
    }
    client.post(LANDING_PAGE_URL,
                data=form_payload,
                headers={'Referer': LANDING_PAGE_URL})

    if 'csrftoken' in client.cookies:
        csrftoken = client.cookies['csrftoken']
    else:
        csrftoken = client.cookies['csrf']
    return csrftoken


def senator_reports(client: requests.Session) -> List[List[str]]:
    """Retrieves all senator reports using pagination.
    Args:
        client: A requests.Session object for making HTTP requests.
    Returns:
        A list of lists of strings, representing the collected reports.
    """
    token = _csrf(client)
    idx = 0
    reports = reports_api(client, idx, token)
    all_reports: List[List[str]] = []
    while len(reports) != 0:
        all_reports.extend(reports)
        idx += BATCH_SIZE
        reports = reports_api(client, idx, token)
    return all_reports


def reports_api(
    client: requests.Session,
    offset: int,
    token: str
) -> List[List[str]]:
    """Queries the periodic transaction reports API.
    Args:
        client: A requests.Session object for making HTTP requests.
        offset: The starting index for the data to retrieve.
        token: The CSRF token for the request.
    Returns:
        A list of lists of strings, representing the transaction reports data.
        Returns an empty list if no data is found.
    """
    login_data = {
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
        'csrfmiddlewaretoken': token
    }
    LOGGER.info('Getting rows starting at {}'.format(offset))
    response = client.post(REPORTS_URL,
                           data=login_data,
                           headers={'Referer': SEARCH_PAGE_URL})
    return response.json()['data']



def _tbody_from_link(client: requests.Session, link: str) -> Optional[Any]:
    """Retrieves the tbody element containing transactions from a given link.
    Args:
        client: A requests.Session object for making HTTP requests.
        link: The relative URL link to the report page.
    Returns:
        The BeautifulSoup object representing the tbody element if found,
        otherwise None.
    """
    report_url = '{0}{1}'.format(ROOT, link)
    report_response = client.get(report_url)
    # If the page is redirected, then the session ID has expired
    if report_response.url == LANDING_PAGE_URL:
        LOGGER.info('Resetting CSRF token and session cookie')
        _csrf(client)
        report_response = client.get(report_url)
    report = BeautifulSoup(report_response.text, 'lxml')
    tbodies = report.find_all('tbody')
    if len(tbodies) == 0:
        return None
    return tbodies[0]


def txs_for_report(client: requests.Session, row: List[str]) -> pd.DataFrame:
    """Converts a row from the periodic transaction reports API to a DataFrame.
    Args:
        client: A requests.Session object for making HTTP requests.
        row: A list of strings representing a row from the reports API.
    Returns:
        A Pandas DataFrame containing the extracted transaction data.  Returns
        an empty DataFrame if the report is a PDF link or if no tbody element
        is found.
    """
    first, last, _, link_html, date_received = row
    link = BeautifulSoup(link_html, 'lxml').a.get('href')
    if link[:len(PDF_PREFIX)] == PDF_PREFIX:
        return pd.DataFrame()

    tbody = _tbody_from_link(client, link)
    if not tbody:
        return pd.DataFrame()

    stocks = []
    for table_row in tbody.find_all('tr'):
        cols = [c.get_text() for c in table_row.find_all('td')]
        tx_date, ticker, asset_name, asset_type, order_type, tx_amount =\
            cols[1], cols[3], cols[4], cols[5], cols[6], cols[7]
        if asset_type != 'Stock' and ticker.strip() in ('--', ''):
            continue
        stocks.append([
            tx_date,
            date_received,
            last,
            first,
            order_type,
            ticker,
            asset_name,
            tx_amount
        ])
    return pd.DataFrame(stocks).rename(
        columns=dict(enumerate(REPORT_COL_NAMES)))



def main() -> pd.DataFrame:
    """Main function to retrieve and compile senator transaction data.
    Returns:
        A Pandas DataFrame containing all extracted senator transaction data.
    """
    LOGGER.info('Initializing client')
    client = requests.Session()
    client.get = add_rate_limit(client.get)
    client.post = add_rate_limit(client.post)
    reports = senator_reports(client)
    all_txs = pd.DataFrame()
    for i, row in enumerate(reports):
        if i % 10 == 0:
            LOGGER.info('Fetching report #{}'.format(i))
            LOGGER.info('{} transactions total'.format(len(all_txs)))
        txs = txs_for_report(client, row)
        all_txs = pd.concat([all_txs, txs], ignore_index=True)
    return all_txs




def upload_embeddings_to_gcs(embeddings, doc_ids, bucket_name, destination_blob_name):
    """Uploads embeddings to Google Cloud Storage.
    Args:
        embeddings: A numpy array of embeddings (each embedding is a list of floats).
        doc_ids: A list of document IDs corresponding to the embeddings.
        bucket_name: The string name of the GCS bucket.
        destination_blob_name: The string name of the blob (file) within the bucket.
    """
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
    """Processes data, creates embeddings, and updates a Vertex AI Vector Search index.
    Args:
        bucket_name: The string name of the GCS bucket.
        filename: The string name of the file to load from the bucket (initial data).
    """
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
    save_to_gcs(GCS_BUCKET_NAME, 'senate_trade.pickle', senator_txs)
    LOGGER.info('Successfully uploaded senate_trade.pickle to GCS bucket: {}'.format(GCS_BUCKET_NAME))

    process_data(GCS_BUCKET_NAME, 'senate_trade.pickle')
