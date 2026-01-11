# main.py (remove vector parts, only build BM25, use bm25_search for retrieval)
import pandas as pd
import time
from utils import process_multiple_documents
from bm25_index import load_or_create_bm25_index, bm25_search
from response_generation import generate_response
import os
from tqdm import tqdm

def build_indices(data_dir="data", force_rebuild=False):
    documents = process_multiple_documents(data_dir)
    
    if force_rebuild:
        if os.path.exists("./bm25_index.pkl"):
            os.remove("./bm25_index.pkl")
    
    bm25_index = load_or_create_bm25_index(documents)
    
    return documents, bm25_index

if __name__ == "__main__":
    # Build indices first (set force_rebuild=True if you want to rebuild every time)
    documents, bm25_index = build_indices(force_rebuild=False)
    
    # Now process queries
    csv_path = r"your csv path"
    queries_df = pd.read_csv(csv_path)
    queries = queries_df["Query"].tolist()
    
    results = []
    total_time = 0.0
    k = 5
    
    # 使用tqdm显示查询处理进度
    for query in tqdm(queries, desc="处理查询", unit="个查询"):
        start_time = time.time()
        
        retrieved_docs = bm25_search(bm25_index, documents, query, k=k)
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
    results_df.to_csv("rag_results.csv", index=False)
    
    print(f"Total processing time: {total_time} seconds")
    print(f"Average time per query: {avg_time} seconds")
    
    with open("summary.txt", "w") as f:
        f.write(f"Total time: {total_time}\n")
        f.write(f"Average time: {avg_time}\n")