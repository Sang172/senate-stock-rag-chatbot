
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import logging
from google.cloud import storage
from google.cloud import aiplatform


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

load_dotenv()
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
REGION = os.environ.get('GCP_REGION')
INDEX_ENDPOINT_ID = os.environ.get('INDEX_ENDPOINT_ID')
DEPLOYED_INDEX_ID = os.environ.get('DEPLOYED_INDEX_ID')


class RAG:
    def __init__(self, documents, doc_embeddings, model_name="gemini-2.0-flash"):
        self.documents = documents
        self.doc_embeddings = doc_embeddings
        self.llm = genai.GenerativeModel(model_name)
        self.memory = []
        self.index_endpoint = self.get_index_endpoint()
    
    def get_index_endpoint(self):
        """Retrieves a Vertex AI Matching Engine Index Endpoint"""
        aiplatform.init(project=PROJECT_ID, location=REGION)
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_ID)
        return index_endpoint

    def get_gemini_response(self, prompt):
        """Returns a parsed response from Gemini given a prompt string"""
        response = self.llm.generate_content(prompt)
        return response.text.strip()
    
    def get_embedding(self, text, model="models/text-embedding-004"):
        """Retrieves vector embedding for a query string"""
        result = genai.embed_content(
            model=model,
            content=text
        )
        return result['embedding']

    def augment_query(self, user_query):
        """Augments user query string for better vector search"""
        prompt = f"""
        You are an expert in information about the stock market.
        Identify all relevant information from the user query (and conversation history if provided) below: person names, time, and company names/stock tickers.
        If you cannot identify any relevant information, just say 'None'.
        
        Examples:
        User Query: find whether or not Vance traded AAPL at any point
        Response: person: Vance, company: Apple (AAPL)

        User Query: find everyone who traded Amazon in May 2021 and May 2022
        Response: company: Amazon (AMZN), time: May 2021, May 2022

        User Query: find who traded any of the magnificent 7 stock in May 2023 
        Response: company: Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), Alphabet/Google (GOOG, GOOGL), Amazon (AMZN), Facebook/Meta (FB/META), Tesla (TSLA), time: May 2023
        
        User Query: {user_query}
        Response:
        """
        info = self.get_gemini_response(prompt)
        return user_query + '\n' + info


    def retrieve_docs(self, user_input, threshold=0.5):
        """Retrieves documents similar to a user input query.
        Args:
            user_input: The user's query string.
            threshold: The minimum cosine similarity score for a document to be
                considered relevant (default: 0.5).
        Returns:
            A list of documents (strings) that are similar to the user input,
            sorted by similarity in descending order.
        """
        input_embedding = self.get_embedding(user_input)
        response = self.index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID, 
            queries=[input_embedding],
            num_neighbors=150
        )
        similar_documents = []
        for neighbor in response[0]:
            i = int(neighbor.id)
            similarity = neighbor.distance
            if similarity >= threshold:
                similar_documents.append([self.documents[i], similarity])
        similar_documents = sorted(similar_documents, key = lambda x: x[1], reverse=True)
        similar_documents = [x[0] for x in similar_documents]
        return similar_documents


    def run(self, user_input):
        """Runs the chatbot query and response generation process.
        Args:
            user_input: The user's query string.
        Returns:
            The LLM's response as a string.
        """

        history_str = ""
        if self.memory:
            history_str = "\n\nConversation History:\n"
            for turn in self.memory:
                history_str += f"User: {turn['user']}\nLLM: {turn['llm']}\n"
        augmented_query = self.augment_query(user_input + history_str)

        retrieved_docs = self.retrieve_docs(augmented_query)
        if not retrieved_docs:
            return "I'm sorry. I do not have the relevant data to answer your question."
        logger.info(f"{len(retrieved_docs)} documents retrieved")
        logger.info(f"{retrieved_docs[0]}")

        prompt = f"""
        You are an expert in analyzing stock transaction records.
        Answer the user query '{user_input}' based on the documents provided below.
        Your response should only be based on the user query and documents.
        Also take the conversation history into account if provided and relevant to the query.
        If there are documents that are unrelated to the query, ignore those documents.
        If none of the documents have information that is necessary to answer the user query, say 'I'm sorry, I do not have the relevant data to answer your question.'.
        If the user query is unrelated to stock trading, say 'I'm sorry, I can only provide answers related to stock trading records.'.
        In your response, do not mention that you obtained information from the documents/data/records.
        Speak as if you already know the information without relying on documents/data/records.

        Documents:

        {'\n\n'.join(retrieved_docs)}
        """
        if self.memory:
            prompt += "\n\nConversation History:\n"
            for turn in self.memory:
                prompt += f"User: {turn['user']}\nLLM: {turn['llm']}\n"

        response = self.get_gemini_response(prompt)

        self.memory.append({"user": user_input, "llm": response})

        if len(self.memory) > 5:
            self.memory.pop(0)

        return response
    

def load_from_gcs(bucket_name, filename):
    storage_client = storage.Client('a')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    with blob.open("rb") as file:
        data = pickle.load(file)
    return data

logger.info("Loading documents from GCS")
documents = load_from_gcs(GCS_BUCKET_NAME, 'documents.pickle')
logger.info("Successfully loaded documentss")

logger.info("Loading embeddings from GCS")
doc_embeddings = load_from_gcs(GCS_BUCKET_NAME, 'doc_embeddings.pickle')
logger.info("Successfully loaded embeddings")

rag = RAG(documents, doc_embeddings)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", messages=rag.memory)


@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if user_input.lower() == 'exit':
        rag.memory = []
        return jsonify({"response": "Conversation cleared."})

    response = rag.run(user_input)
    return jsonify({"response": response, "history": rag.memory}) 

if __name__ == "__main__":
    logger.info("About to start Flask app on port 5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)