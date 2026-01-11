import numpy as np
from utils import get_embedding_model
from summary_index import summary_search
from detail_index import detail_search
from bm25_index import bm25_search

def hierarchical_retrieval(query, summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index, top_k_summary=5, top_k_detail=5, alpha=0.8):
    print(f"Performing hierarchical retrieval for query: {query}")
    epsilon = 1e-8
    
    # First stage: Retrieve relevant summaries (pages) using vector search
    relevant_summaries = summary_search(summary_index, summary_texts, summary_metas, query, k=top_k_summary)
    
    # Collect page filters (e.g., by source and page)
    page_filters = [{"source": sum["metadata"]["source"], "page": sum["metadata"]["page"]} for sum in relevant_summaries]
    
    # Second stage: Hybrid retrieval (vector + BM25) within filtered pages
    retrieved_docs = []
    for filter_meta in page_filters:
        # Vector search
        vector_results = detail_search(detail_index, detail_texts, detail_metas, query, k=top_k_detail * 2, filter_metadata=filter_meta)
        # BM25 search
        bm25_results = bm25_search(bm25_index, detail_texts, detail_metas, query, k=top_k_detail * 2)
        
        # Combine results
        combined_results = []
        seen_indices = set()
        
        # Process vector results
        for result in vector_results:
            idx = detail_texts.index(result["text"])
            if idx not in seen_indices and (filter_meta is None or (result["metadata"].get("source") == filter_meta.get("source") and result["metadata"].get("page") == filter_meta.get("page"))):
                combined_results.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": result["similarity"],
                    "bm25_score": 0.0,
                    "index": idx
                })
                seen_indices.add(idx)
        
        # Process BM25 results
        for result in bm25_results:
            idx = detail_texts.index(result["text"])
            if idx not in seen_indices and (filter_meta is None or (result["metadata"].get("source") == filter_meta.get("source") and result["metadata"].get("page") == filter_meta.get("page"))):
                combined_results.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": result["bm25_score"],
                    "index": idx
                })
                seen_indices.add(idx)
            elif idx in seen_indices:
                # Update BM25 score for existing vector result
                for res in combined_results:
                    if res["index"] == idx:
                        res["bm25_score"] = result["bm25_score"]
                        break
        
        # Normalize and combine scores
        vector_scores = np.array([doc["vector_score"] for doc in combined_results])
        bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
        
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
        retrieved_docs.extend(combined_results[:top_k_detail])
    
    # Sort by combined score and take top_k_detail overall
    retrieved_docs.sort(key=lambda x: x["combined_score"], reverse=True)
    top_docs = retrieved_docs[:top_k_detail]
    
    print(f"Retrieved {len(top_docs)} documents with hierarchical retrieval")
    return top_docs