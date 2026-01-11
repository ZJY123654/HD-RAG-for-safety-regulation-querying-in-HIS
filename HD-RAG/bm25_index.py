from rank_bm25 import BM25Okapi
import pickle
import os

def create_bm25_index(chunks_text):
    tokenized_docs = [text.split() for text in chunks_text]
    bm25 = BM25Okapi(tokenized_docs)
    print(f"Created BM25 index with {len(chunks_text)} documents")
    return bm25

def load_or_create_bm25_index(chunks_text, persist_path="./bm25_index.pkl"):
    if os.path.exists(persist_path):
        with open(persist_path, "rb") as f:
            bm25 = pickle.load(f)
        print(f"Loaded existing BM25 index from {persist_path}")
    else:
        bm25 = create_bm25_index(chunks_text)
        with open(persist_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"Saved BM25 index to {persist_path}")
    return bm25

def bm25_search(bm25, documents_text, metadatas, query, k=5):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    results = []
    for i, score in enumerate(scores):
        results.append({
            "text": documents_text[i],
            "metadata": metadatas[i],
            "bm25_score": float(score)
        })
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:k]