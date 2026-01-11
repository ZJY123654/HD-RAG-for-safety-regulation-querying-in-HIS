import numpy as np
from utils import get_embedding_model
from vector_store import similarity_search_with_scores, get_all_documents
from bm25_index import bm25_search

def fusion_retrieval(query, documents, vector_collection, bm25_index, k=5, alpha=0.8):
    print(f"Performing fusion retrieval for query: {query}")
    epsilon = 1e-8
    
    # Pass the original query string to similarity_search_with_scores instead of the embedding
    vector_results = similarity_search_with_scores(vector_collection, query, k=len(documents))
    
    bm25_results = bm25_search(bm25_index, documents, query, k=len(documents))
    
    vector_scores_dict = {result["metadata"].get("index", i): result["similarity"] for i, result in enumerate(vector_results)}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    all_docs = get_all_documents(vector_collection)
    combined_results = []
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # Handle empty arrays and division by zero
    if len(vector_scores) > 0:
        vector_min, vector_max = np.min(vector_scores), np.max(vector_scores)
        if vector_max - vector_min > 0:
            norm_vector_scores = (vector_scores - vector_min) / (vector_max - vector_min + epsilon)
        else:
            norm_vector_scores = np.zeros_like(vector_scores)
    else:
        norm_vector_scores = np.array([])
    
    if len(bm25_scores) > 0:
        bm25_min, bm25_max = np.min(bm25_scores), np.max(bm25_scores)
        if bm25_max - bm25_min > 0:
            norm_bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min + epsilon)
        else:
            norm_bm25_scores = np.zeros_like(bm25_scores)
    else:
        norm_bm25_scores = np.array([])
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    top_results = combined_results[:k]
    
    print(f"Retrieved {len(top_results)} documents with fusion retrieval")
    return top_results