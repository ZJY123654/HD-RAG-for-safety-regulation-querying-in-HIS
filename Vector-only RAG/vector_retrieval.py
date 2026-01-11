from utils import get_embedding_model
from vector_store import similarity_search_with_scores

def vector_retrieval(query, vector_collection, k=5):
    print(f"Performing vector retrieval for query: {query}")
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    top_results = similarity_search_with_scores(vector_collection, query, k=k)
    print(f"Retrieved {len(top_results)} documents with vector retrieval")
    return top_results