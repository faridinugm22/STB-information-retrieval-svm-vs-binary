from flask import Flask, request, render_template
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import numpy as np
import time

app = Flask(__name__)

client = MongoClient('mongodb://localhost:27017/')
db = client['stbi']
collection = db['stbi']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def fetch_documents():
    return list(collection.find())

def compute_tfidf(documents):
    texts = [f"{doc['DAPIL']} {doc['PROVINSI']} {doc['NAMA_DAPIL']} {doc['NOMOR PARTAI']} {doc['NAMA PARTAI']} {doc['NOMOR URUT']} {doc['NAMA CALEG']} {doc['JENIS_KELAMIN']}" for doc in documents]
    texts = [preprocess_text(text) for text in texts]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def binary_search(query_vector, tfidf_matrix):
    binary_matrix = (tfidf_matrix > 0).astype(int)  # Convert TF-IDF to binary representation
    query_binary = (query_vector > 0).astype(int)
    similarities = np.dot(query_binary, binary_matrix.T).toarray().flatten()  # Dot product for binary matching
    return similarities

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    precision = recall = search_time = num_results = 0  # Initialize num_results
    method = 'vsm'  # Default method
    displayed_results = None  # Initialize displayed_results
    relevant_docs = 0  # Initialize relevant_docs
    retrieved_relevant_docs = 0  # Initialize retrieved_relevant_docs

    if request.method == 'POST':
        start_time = time.time()
        query = request.form.get('query', '')
        method = request.form.get('method', 'vsm')

        if query:
            documents = fetch_documents()
            tfidf_matrix, vectorizer = compute_tfidf(documents)
            query_processed = preprocess_text(query)
            query_vector = vectorizer.transform([query_processed])

            similarities = linear_kernel(query_vector, tfidf_matrix).flatten() if method == 'vsm' else binary_search(query_vector, tfidf_matrix)

            filtered_results = [(i, score) for i, score in enumerate(similarities) if score > 0]
            top_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
            results = [(documents[i], score) for i, score in top_results]

            num_results = len(results)  # Calculate the total number of results

            # tampilkan semua hasil
            displayed_results = results[:num_results]

            # Calculate precision and recall for the selected method
            relevant_docs = len([doc for doc in documents if query.lower() in doc['NAMA CALEG'].lower()])
            retrieved_relevant_docs = len([r for r, _ in displayed_results if query.lower() in r['NAMA CALEG'].lower()])
            precision = retrieved_relevant_docs / len(displayed_results) if displayed_results else 0
            recall = retrieved_relevant_docs / relevant_docs if relevant_docs > 0 else 0

        search_time = time.time() - start_time

    return render_template('combined_interface_pagenation.html', query=query if request.method == 'POST' else "", results=displayed_results, method=method, relevant_docs= relevant_docs, retrieved_relevant_docs= retrieved_relevant_docs, precision=precision, recall=recall, search_time=search_time, num_results=num_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
