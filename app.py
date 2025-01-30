import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify
import boto3
import logging
from annoy import AnnoyIndex


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)




load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def lambda_handler(event, context):
    # Your code here
    return {
        'statusCode': 200,
        'body': 'Hello from Lambda!'
    }

class RAG:
    def __init__(self, documents, doc_embeddings, model_name="gemini-1.5-flash-latest"):
        self.documents = documents
        self.doc_embeddings = np.array(doc_embeddings)
        self.llm = genai.GenerativeModel(model_name)
        self.memory = []
        self.index = self.create_index(self.doc_embeddings)

    def create_index(self, embeddings):
        dimension = embeddings.shape[1]
        index = AnnoyIndex(dimension, 'euclidean')
        for i, emb in enumerate(embeddings):
            index.add_item(i, emb)
        index.build(200)
        return index

    def get_embedding(self, text, model="models/text-embedding-004"):
        result = genai.embed_content(
            model=model,
            content=text
        )
        return result['embedding']

    def retrieve_docs(self, user_input, threshold=0.5):
        input_embedding = self.get_embedding(user_input)
        similar_indices = self.index.get_nns_by_vector(input_embedding, 150, include_distances=False)
        similar_documents = []
        for i in similar_indices:
            similarity = cosine_similarity([input_embedding], [self.doc_embeddings[i]])
            if similarity >= threshold:
                similar_documents.append([self.documents[i], similarity])
        similar_documents = sorted(similar_documents, key = lambda x: x[1], reverse=True)
        similar_documents = [x[0] for x in similar_documents]
        return similar_documents

    def get_gemini_response(self, prompt):
        response = self.llm.generate_content(prompt)
        return response.text.strip()
    
    def augment_query(self, user_query):
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
        Response: company: Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), Alphabet (GOOG, GOOGL), Amazon (AMZN), Facebook/Meta (FB/META), Tesla (TSLA), time: May 2023
        
        User Query: {user_query}
        Response:
        """
        info = self.get_gemini_response(prompt)
        return user_query + '\n' + info


    def run(self, user_input):

        history_str = ""
        if self.memory:
            history_str = "\n\nConversation History:\n"
            for turn in self.memory:
                history_str += f"User: {turn['user']}\nLLM: {turn['llm']}\n"
        augmented_query = self.augment_query(user_input + history_str)
        # print(augmented_query + '\n')

        retrieved_docs = self.retrieve_docs(augmented_query)
        if not retrieved_docs:
            return "I'm sorry. I have no data that is relevant to your input."
        # print(f"{len(retrieved_docs)} documents retrieved")
        # print(retrieved_docs[0])

        prompt = "You are an expert in analyzing stock transaction records."
        prompt += f"\nAnswer the user query '{user_input}' based on the documents provided below."
        prompt += "\nAlso take the conversation history into account if provided and relevant."
        prompt += "\nIf there are documents that are irrelevant to the query, ignore those documents."
        prompt += "\nIn your response, do not mention that you obtained information from the documents/data/records."
        prompt += "\nSpeak as if you already know the information without relying on documents/data/records."
        prompt += '\n\nDocuments:\n'
        prompt += '\n\n'.join(retrieved_docs)
        if self.memory:
            prompt += "\n\nConversation History:\n"
            for turn in self.memory:
                prompt += f"User: {turn['user']}\nLLM: {turn['llm']}\n"

        response = self.get_gemini_response(prompt)

        self.memory.append({"user": user_input, "llm": response})

        if len(self.memory) > 5:
            self.memory.pop(0)

        return response
    


app = Flask(__name__)


S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
s3 = boto3.client('s3')

logger.info("Loading documents from S3")
s3.download_file(S3_BUCKET_NAME, 'documents.pickle', 'documents.pickle')
with open('documents.pickle', 'rb') as file:
    documents = pickle.load(file)
logger.info("Successfully loaded documents")

logger.info("Loading embeddings from S3")
s3.download_file(S3_BUCKET_NAME, 'doc_embeddings.pickle', 'doc_embeddings.pickle')
with open('doc_embeddings.pickle', 'rb') as file:
    doc_embeddings = pickle.load(file)
logger.info("Successfully loaded embeddings")

rag = RAG(documents, doc_embeddings)

@app.route("/")
def index():
    return render_template("index.html", messages=rag.memory)

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if user_input.lower() == 'exit':
        rag.memory = []
        return jsonify({"response": "Conversation cleared."})  # or handle differently

    response = rag.run(user_input)
    return jsonify({"response": response, "history": rag.memory})  # Include history

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False for production