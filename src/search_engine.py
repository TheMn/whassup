from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.thread_texts = None
        self.threads = None

    def build_index(self, threads):
        """Builds the TF-IDF matrix from the message threads."""
        self.threads = threads
        self.thread_texts = [" ".join([m["text"] for m in thread]) for thread in threads.values()]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.thread_texts)

    def search(self, query, k=3):
        """Performs a search for the given query."""
        if self.tfidf_matrix is None:
            raise Exception("Index has not been built yet. Call build_index() first.")

        query_vector = self.vectorizer.transform([query])

        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get the top k most similar threads
        # We use argpartition to get the top k without a full sort
        # We need to handle cases where k is larger than the number of documents
        num_docs = self.tfidf_matrix.shape[0]
        if k > num_docs:
            k = num_docs

        # We want the top k largest values, so we negate the array and find the smallest
        top_k_indices = np.argpartition(-cosine_similarities, range(k))[:k]

        # Sort the top k indices by similarity score
        top_k_indices = top_k_indices[np.argsort(-cosine_similarities[top_k_indices])]

        results = []
        thread_keys = list(self.threads.keys())
        for i in top_k_indices:
            # Filter out results with 0 similarity
            if cosine_similarities[i] > 0:
                thread_id = thread_keys[i]
                thread_messages = self.threads[thread_id]

                # Find the most relevant message in the thread.
                # For simplicity, we can return the first message,
                # or we can try to find the message that is most similar to the query.
                # Let's find the most similar message within the thread.

                message_vectors = self.vectorizer.transform([msg['text'] for msg in thread_messages])
                message_similarities = cosine_similarity(query_vector, message_vectors).flatten()
                best_message_index = np.argmax(message_similarities)
                best_message = thread_messages[best_message_index]

                results.append({
                    "message_id": best_message['id'],
                    "text": best_message['text']
                })
        return results
