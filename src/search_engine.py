import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

CACHE_DIR = ".cache"
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "thread_embeddings.pkl")
THREADS_PATH = os.path.join(CACHE_DIR, "threads.pkl")

class SearchEngine:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
        self.thread_embeddings = None
        self.threads = None
        if os.path.exists(CACHE_DIR):
            self.load_index()
        else:
            os.makedirs(CACHE_DIR)

    def build_index(self, threads):
        """Builds the sentence embeddings for the message threads and saves them to cache."""
        self.threads = threads
        thread_texts = [" ".join([m["text"] for m in thread]) for thread in threads.values()]

        print("Generating embeddings for threads...")
        self.thread_embeddings = self.model.encode(thread_texts, convert_to_tensor=True, show_progress_bar=True)
        print("Embeddings generated.")

        self.save_index()

    def save_index(self):
        """Saves the thread_embeddings and threads to cache."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.thread_embeddings, f)
        with open(THREADS_PATH, "wb") as f:
            pickle.dump(self.threads, f)

    def load_index(self):
        """Loads the thread_embeddings and threads from cache."""
        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(THREADS_PATH):
            print("Loading index from cache...")
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.thread_embeddings = pickle.load(f)
            with open(THREADS_PATH, "rb") as f:
                self.threads = pickle.load(f)
            print("Index loaded.")

    def has_index(self):
        """Checks if a cached index exists."""
        return self.thread_embeddings is not None and self.threads is not None

    def search(self, query, k=15):
        """Performs a semantic search for the given query."""
        if not self.has_index():
            raise Exception("Index has not been built or loaded yet.")

        query_embedding = self.model.encode(query, convert_to_tensor=True)

        cosine_similarities = util.cos_sim(query_embedding, self.thread_embeddings)[0]
        cosine_similarities = cosine_similarities.cpu()

        num_docs = self.thread_embeddings.shape[0]
        if k > num_docs:
            k = num_docs

        # Use torch for topk for simplicity
        top_k = torch.topk(cosine_similarities, k=k)
        top_k_indices = top_k.indices
        top_k_values = top_k.values

        results = []
        thread_keys = list(self.threads.keys())
        for i, score in zip(top_k_indices, top_k_values):
            if score > 0.3: # Threshold to filter out irrelevant results
                thread_id = thread_keys[i]
                thread_messages = self.threads[thread_id]

                message_texts = [msg['text'] for msg in thread_messages]
                if not message_texts:
                    continue

                message_embeddings = self.model.encode(message_texts, convert_to_tensor=True)
                message_similarities = util.cos_sim(query_embedding, message_embeddings)[0].cpu()
                best_message_index = torch.argmax(message_similarities)
                best_message = thread_messages[best_message_index]

                results.append({
                    "message_id": best_message['id'],
                    "text": best_message['text'],
                    "from": best_message.get('from', 'Unknown Sender'),
                    "date": best_message.get('date')
                })
        return results
