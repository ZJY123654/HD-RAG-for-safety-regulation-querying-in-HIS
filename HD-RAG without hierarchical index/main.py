import pandas as pd
import time
from tqdm import tqdm
from utils import process_multiple_documents
from vector_store import load_or_create_vector_store
from bm25_index import load_or_create_bm25_index
from fusion_retrieval import fusion_retrieval
from response_generation import generate_response

if __name__ == "__main__":
    data_dir = "data"  # Directory with multiple files (md, docx, pdf)
    documents = process_multiple_documents(data_dir)
    
    vector_collection = load_or_create_vector_store(documents)
    bm25_index = load_or_create_bm25_index(documents)
    
    # Load queries from CSV
    csv_path = "Customized test dataset.csv"
    queries_df = pd.read_csv(csv_path)
    queries = queries_df["Query"].tolist()
    
    results = []
    total_time = 0.0
    k = 5
    alpha = 0.8
    
    for query in tqdm(queries, desc="处理查询", unit="个"):
        start_time = time.time()
        
        retrieved_docs = fusion_retrieval(query, documents, vector_collection, bm25_index, k=k, alpha=alpha)
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
        
        # 显示单个查询的处理时间
        tqdm.write(f"查询 '{query[:30]}...' 处理时间: {query_time:.4f}秒")
    
    avg_time = total_time / len(queries) if queries else 0.0
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("rag_results.csv", index=False)
    
    # Print summary
    print(f"Total processing time: {total_time} seconds")
    print(f"Average time per query: {avg_time} seconds")
    
    # Optionally save summary
    with open("summary.txt", "w") as f:
        f.write(f"Total time: {total_time}\n")
        f.write(f"Average time: {avg_time}\n")