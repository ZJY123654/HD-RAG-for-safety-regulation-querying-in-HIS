import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm
from utils import get_embedding_model, chunk_text
from bm25_index import load_or_create_bm25_index

def load_or_create_detail_index(pages, index_path="./faiss_detail_index.index", data_path="./faiss_detail_data.pkl", bm25_path="./bm25_index.pkl"):
    if os.path.exists(index_path) and os.path.exists(data_path) and os.path.exists(bm25_path):
        index = faiss.read_index(index_path)
        with open(data_path, 'rb') as f:
            documents_text, metadatas = pickle.load(f)
        # 赋值给函数返回的变量名
        all_chunks_text = documents_text
        all_metadatas = metadatas
        bm25_index = load_or_create_bm25_index(documents_text, bm25_path)
        print(f"Loaded existing Faiss detail index and BM25 index")
    else:
        embedding_model = get_embedding_model()
        all_chunks_text = []
        all_metadatas = []
        for page in tqdm(pages, desc="文本分块", unit="页"):
            chunks = chunk_text(page["text"], page["metadata"])
            for chunk in chunks:
                all_chunks_text.append(chunk.page_content)
                all_metadatas.append(chunk.metadata)
        
        embeddings = embedding_model.embed_documents(all_chunks_text)
        emb_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(emb_array)
        
        dimension = emb_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(emb_array)
        
        bm25_index = load_or_create_bm25_index(all_chunks_text, bm25_path)
        
        faiss.write_index(index, index_path)
        with open(data_path, 'wb') as f:
            pickle.dump((all_chunks_text, all_metadatas), f)
        print(f"Created and saved Faiss detail index with {len(all_chunks_text)} chunks and BM25 index")
    
    return index, all_chunks_text, all_metadatas, bm25_index

def detail_search(index, documents_text, metadatas, query, k=5, filter_metadata=None):
    """
    在详细索引中搜索相关文档
    
    参数:
    - index: Faiss索引
    - documents_text: 文档文本列表
    - metadatas: 文档元数据列表
    - query: 查询文本
    - k: 返回的结果数量
    - filter_metadata: 过滤条件，字典格式，例如{"source": "file.pdf", "page": 5}
    
    返回:
    - 搜索结果列表
    """
    embedding_model = get_embedding_model()
    query_embedding = np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k * 2)  # 搜索更多结果以便过滤
    
    output = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1:
            # 如果有过滤条件，检查是否匹配
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if metadatas[idx].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            output.append({
                "text": documents_text[idx],
                "metadata": metadatas[idx],
                "similarity": D[0][i]
            })
            
            # 收集足够的结果后停止
            if len(output) >= k:
                break
    
    return output