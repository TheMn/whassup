import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CACHE_DIR = ".cache"
VECTORIZER_PATH = os.path.join(CACHE_DIR, "vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
THREADS_PATH = os.path.join(CACHE_DIR, "threads.pkl")

class SearchEngine:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.threads = None
        self.thread_texts = None
        if os.path.exists(CACHE_DIR):
            self.load_index()
        else:
            os.makedirs(CACHE_DIR)

    def build_index(self, threads):
        """Builds the TF-IDF matrix from the message threads and saves it to cache."""
        self.threads = threads
        self.thread_texts = [" ".join([m["text"] for m in thread]) for thread in threads.values()]

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.thread_texts)

        self.save_index()

    def save_index(self):
        """Saves the vectorizer, tfidf_matrix, and threads to cache."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(TFIDF_MATRIX_PATH, "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(THREADS_PATH, "wb") as f:
            pickle.dump(self.threads, f)

    def load_index(self):
        """Loads the vectorizer, tfidf_matrix, and threads from cache."""
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(THREADS_PATH):
            with open(VECTORIZER_PATH, "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(TFIDF_MATRIX_PATH, "rb") as f:
                self.tfidf_matrix = pickle.load(f)
            with open(THREADS_PATH, "rb") as f:
                self.threads = pickle.load(f)

    def has_index(self):
        """Checks if a cached index exists."""
        return self.vectorizer is not None and self.tfidf_matrix is not None and self.threads is not None

    def search(self, query, k=3):
        """Performs a search for the given query."""
        if not self.has_index():
            raise Exception("Index has not been built or loaded yet.")

        query_vector = self.vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        num_docs = self.tfidf_matrix.shape[0]
        if k > num_docs:
            k = num_docs

        top_k_indices = np.argpartition(-cosine_similarities, range(k))[:k]
        top_k_indices = top_k_indices[np.argsort(-cosine_similarities[top_k_indices])]

        results = []
        thread_keys = list(self.threads.keys())
        for i in top_k_indices:
            if cosine_similarities[i] > 0:
                thread_id = thread_keys[i]
                thread_messages = self.threads[thread_id]

                message_vectors = self.vectorizer.transform([msg['text'] for msg in thread_messages])
                message_similarities = cosine_similarity(query_vector, message_vectors).flatten()
                best_message_index = np.argmax(message_similarities)
                best_message = thread_messages[best_message_index]

                results.append({
                    "message_id": best_message['id'],
                    "text": best_message['text'],
                    "from": best_message.get('from', 'Unknown Sender'),
                    "date": best_message.get('date')
                })
        return results
