import pandas as pd
import time
from utils import process_multiple_documents
from summary_index import load_or_create_summary_index
from detail_index import load_or_create_detail_index
from hierarchical_retrieval import hierarchical_retrieval
from response_generation import generate_response
import os
import shutil
from tqdm import tqdm

def build_indices(data_dir="data", force_rebuild=False):
    pages = process_multiple_documents(data_dir)
    
    # FAISS索引存储在faiss_indexes目录下，删除相关文件
    if force_rebuild:
        faiss_dir = "./faiss_indexes"
        if os.path.exists(faiss_dir):
            # 删除summary_index相关文件
            for file in ["summary_index_index.faiss", "summary_index_metadata.pkl", "summary_index_ids.pkl", "summary_index_documents.pkl"]:
                file_path = os.path.join(faiss_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed {file_path}")
            
            # 删除detail_index相关文件
            for file in ["detail_index_index.faiss", "detail_index_metadata.pkl", "detail_index_ids.pkl", "detail_index_documents.pkl"]:
                file_path = os.path.join(faiss_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed {file_path}")
    
    summary_collection = load_or_create_summary_index(pages)
    detail_collection = load_or_create_detail_index(pages)
    
    return summary_collection, detail_collection

if __name__ == "__main__":
    # Build indices first (set force_rebuild=True to rebuild)
    summary_collection, detail_collection = build_indices(force_rebuild=False)
    
    # Load queries from CSV
    csv_path = r"your csv path"
    queries_df = pd.read_csv(csv_path)
    queries = queries_df["Query"].tolist()
    
    results = []
    total_time = 0.0
    k_summary = 5
    k_detail = 5
    
    for query in tqdm(queries, desc="Processing queries"):
        start_time = time.time()
        
        retrieved_docs = hierarchical_retrieval(query, summary_collection, detail_collection, top_k_summary=k_summary, top_k_detail=k_detail)
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