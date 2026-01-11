import pandas as pd
import time
from utils import process_multiple_documents
from vector_store import load_or_create_vector_store
from vector_retrieval import vector_retrieval
from response_generation import generate_response
import os
import shutil

def build_indices(data_dir="data", force_rebuild=False):
    documents = process_multiple_documents(data_dir)
    
    # FAISS的默认持久化目录是"./faiss_db"
    if force_rebuild:
        if os.path.exists("./faiss_db"):
            shutil.rmtree("./faiss_db")
    
    vector_collection = load_or_create_vector_store(documents)
    
    return documents, vector_collection

if __name__ == "__main__":
    # Build indices first (set force_rebuild=True if you want to rebuild every time)
    documents, vector_collection = build_indices(force_rebuild=False)
    
    # Now process queries
    csv_path = r"your csv path"
    queries_df = pd.read_csv(csv_path)
    queries = queries_df["Query"].tolist()
    
    results = []
    total_time = 0.0
    k = 5
    
    from tqdm import tqdm
    
    for query in tqdm(queries, desc="处理查询", unit="个"):
        start_time = time.time()
        
        retrieved_docs = vector_retrieval(query, vector_collection, k=k)
        context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
        
        response = generate_response(query, context)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        results.append({
            "Query": query,
            "Retrievaled_context": context,
            "Response": response,
            "Time": query_time
        })
    
    avg_time = total_time / len(queries) if queries else 0.0
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("rag_results.csv", index=False, escapechar='\\')
    
    print(f"Total processing time: {total_time} seconds")
    print(f"Average time per query: {avg_time} seconds")
    
    with open("summary.txt", "w") as f:
        f.write(f"Total time: {total_time}\n")
        f.write(f"Average time: {avg_time}\n")