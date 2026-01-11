import pandas as pd
import time
from tqdm import tqdm
from utils import process_multiple_documents
from summary_index import load_or_create_summary_index
from detail_index import load_or_create_detail_index
from hierarchical_retrieval import hierarchical_retrieval
from response_generation import generate_response
import os

def build_indices(data_dir="data", force_rebuild=False):
    pages = process_multiple_documents(data_dir)
    
    if force_rebuild:
        for path in ["./faiss_summary_index.index", "./faiss_summary_data.pkl", "./faiss_detail_index.index", "./faiss_detail_data.pkl", "./bm25_index.pkl"]:
            if os.path.exists(path):
                os.remove(path)
    
    summary_index, summary_texts, summary_metas = load_or_create_summary_index(pages)
    detail_index, detail_texts, detail_metas, bm25_index = load_or_create_detail_index(pages)
    
    return summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index

if __name__ == "__main__":
    # Build indices first (set force_rebuild=True to rebuild)
    summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index = build_indices(force_rebuild=False)
    
    # Load queries from CSV
    csv_path = r"your csv path"
    queries_df = pd.read_csv(csv_path)
    queries = queries_df["Query"].tolist()
    
    results = []
    total_time = 0.0
    k_summary = 5
    k_detail = 5
    alpha = 0.8
    
    for query in tqdm(queries, desc="处理查询", unit="查询"):
        start_time = time.time()
        
        retrieved_docs = hierarchical_retrieval(query, summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index, top_k_summary=k_summary, top_k_detail=k_detail, alpha=alpha)
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
    # 添加escapechar参数以处理包含特殊字符的数据
    results_df.to_csv("rag_results.csv", index=False, escapechar='\\')
    
    print(f"Total processing time: {total_time} seconds")
    print(f"Average time per query: {avg_time} seconds")
    
    with open("summary.txt", "w") as f:
        f.write(f"Total time: {total_time}\n")
        f.write(f"Average time: {avg_time}\n")