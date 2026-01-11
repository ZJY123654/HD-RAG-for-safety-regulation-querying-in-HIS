from utils import get_embedding_model
from summary_index import summary_search
from detail_index import detail_search

def hierarchical_retrieval(query, summary_collection, detail_collection, top_k_summary=5, top_k_detail=5):
    print(f"Performing hierarchical retrieval for query: {query}")
    
    # First stage: Retrieve relevant summaries (pages)
    relevant_summaries = summary_search(summary_collection, query, k=top_k_summary)
    
    # Collect page filters (e.g., by source and page)
    page_filters = [{"source": sum["metadata"]["source"], "page": sum["metadata"]["page"]} for sum in relevant_summaries]
    
    # Second stage: Retrieve details within those pages
    retrieved_docs = []
    for filter_meta in page_filters:
        details = detail_search(detail_collection, query, k=top_k_detail, filter_metadata=filter_meta)
        retrieved_docs.extend(details)
    
    # Sort by similarity and take top_k_detail overall
    retrieved_docs.sort(key=lambda x: x["similarity"], reverse=True)
    top_docs = retrieved_docs[:top_k_detail]
    
    print(f"Retrieved {len(top_docs)} documents with hierarchical retrieval")
    return top_docs