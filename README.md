# Senate Stock RAG Chatbot

## Overview
This project aims to make US Senate stock trading information more accessible to the public. It uses a Retrieval-Augmented Generation (RAG) chatbot powered by Google Cloud Platform (Cloud Run, Vertex AI, GCS) and Gemini 2.0 Flash to allow users to easily query and analyze senators' stock trades.

## Problem Statement
While the US Senate Public Financial Disclosure Database provides transparency into senators' stock trading activities, it's not user-friendly. The limited search functionality and lack of aggregation features make it difficult to extract meaningful insights.

## Solution
This project addresses the problem by through system that:

- Fetches raw data from the Senate financial disclosure website, cleans it, and transforms it into a list of documents.
- Converts each document list into a vector database using Vertex AI.
- Uses the vector database and Gemini 2.0 Flash to build a RAG chatbot
- Automates the database update on a weekly interval

## Project Structure
The project is organized into three key parts:

- **Root Directory**: Contains the code for the chatbot application, including the main application logic (`app.py`) and the user interface (`templates/index.html`).
- **update/**: Handles the data pipeline, including scraping raw data (`scrape.py`), processing it (`process.py`), generating embeddings, and updating the Vector Search Index.
- **trigger/**: Automates the update process using Google Cloud Scheduler.

## Deployment
The project is containerized using Docker and deployed to Google Cloud Run. A Continuous Integration and Continuous Deployment (CI/CD) pipeline using GitHub Actions automates the build, test, and deployment process.

## Future Development
Plans include:

- Expanding coverage to include the U.S. House of Representatives' financial disclosures.
- Implementing Optical Character Recognition (OCR) and advanced natural language processing techniques to handle unstructured data sources.

## Links
- [Medium Blog Post](https://medium.com/@sang.ahn.94/making-senators-stock-market-trading-information-accessible-a-rag-chatbot-powered-by-vertex-ai-d88f345ddbfc)
- [Demo Video](https://youtu.be/od7znGNRQoA)

