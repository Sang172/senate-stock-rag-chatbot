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
from process import process, get_embedding, load_from_gcs

load_dotenv()
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

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
    def with_rate_limit(*args, **kw):
        time.sleep(RATE_LIMIT_SECS)
        return f(*args, **kw)
    return with_rate_limit


def _csrf(client: requests.Session) -> str:
    """ Set the session ID and return the CSRF token for this session. """
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
    """ Return all results from the periodic transaction reports API. """
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
    """ Query the periodic transaction reports API. """
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
    """
    Return the tbody element containing transactions for this senator.
    Return None if no such tbody element exists.
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
    """
    Convert a row from the periodic transaction reports API to a DataFrame
    of transactions.
    """
    first, last, _, link_html, date_received = row
    link = BeautifulSoup(link_html, 'lxml').a.get('href')
    # We cannot parse PDFs
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


def save_to_gcs(bucket_name, filename, dataframe):
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    temp_file_path = f"temp_{filename}"
    with open(temp_file_path, "wb") as temp_file:
        pickle.dump(dataframe, temp_file)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)


def process_data(bucket_name, filename):
    LOGGER.info("Reading data from senate_trade.pickle")
    data = load_from_gcs(bucket_name, filename)
    LOGGER.info("Successfully read data from senate_trade.pickle")

    LOGGER.info("Processing data")
    documents = process(data)
    LOGGER.info("Data processing complete")

    doc_embeddings = []
    LOGGER.info(f"Start creating vector embeddings for {len(documents)} documents")
    for i, doc in enumerate(documents):
        if (i+1)%100==0:
            LOGGER.info(f"Embedding created for {i+1} th document")
        doc_embeddings.append(get_embedding(doc))
    LOGGER.info("Embedding creation complete")

    doc_embeddings = np.array(doc_embeddings)

    LOGGER.info("Saving documents to documents.pickle in GCS bucket")
    save_to_gcs(bucket_name, 'documents.pickle', documents)
    LOGGER.info("Successfully saved documents to documents.pickle in GCS bucket")

    LOGGER.info("Saving document embeddings to doc_embeddings.pickle in GCS bucket")
    save_to_gcs(bucket_name, 'doc_embeddings.pickle', doc_embeddings)
    LOGGER.info("Successfully saved document embeddings to doc_embeddings.pickle in GCS bucket")

    LOGGER.info("Data processed and saved as pickle files in GCS.")


if __name__ == '__main__':
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    senator_txs = main()
    LOGGER.info('Dumping to .pickle')

    
    save_to_gcs(GCS_BUCKET_NAME, 'senate_trade.pickle', senator_txs)
    LOGGER.info('Successfully uploaded senate_trade.pickle to GCS bucket: {}'.format(GCS_BUCKET_NAME))

    process_data(GCS_BUCKET_NAME, 'senate_trade.pickle')
