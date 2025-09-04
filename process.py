import json

with open("result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

messages = []
for msg in data["messages"]:
    if msg["type"] == "message" and "text" in msg:
        messages.append({
            "id": msg["id"],
            "text": msg["text"] if isinstance(msg["text"], str) else "".join([t if isinstance(t, str) else "" for t in msg["text"]]),
            "reply_to": msg.get("reply_to_message_id"),
        })


from collections import defaultdict

threads = defaultdict(list)
for msg in messages:
    root = msg["reply_to"] or msg["id"]
    threads[root].append(msg)


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = [" ".join([m["text"] for m in thread]) for thread in threads.values()]
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

import faiss
import numpy as np

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings, dtype="float32"))

def search(query, k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_emb, dtype="float32"), k)
    results = []
    for idx in I[0]:
        if idx == -1: continue
        results.append(texts[idx])
    return results

import re

def extract_professor(text):
    match = re.search(r"(استاد\s+\S+|Professor\s+\S+)", text, re.IGNORECASE)
    return match.group(0) if match else None
