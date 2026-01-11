from langchain.schema import Document
from utils import get_embedding_model, generate_summary
from tqdm import tqdm
import faiss
import numpy as np
import os
import pickle

def load_or_create_summary_index(pages, persist_directory="./faiss_indexes", collection_name="summary_index"):
    embedding_model = get_embedding_model()
    
    # 构建索引文件路径
    index_path = os.path.join(persist_directory, f"{collection_name}_index.faiss")
    metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.pkl")
    ids_path = os.path.join(persist_directory, f"{collection_name}_ids.pkl")
    documents_path = os.path.join(persist_directory, f"{collection_name}_documents.pkl")
    
    # 检查是否存在现有的索引文件
    if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(ids_path) and os.path.exists(documents_path):
        print(f"Loading existing FAISS summary index: {collection_name}")
        
        # 加载索引和元数据
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadatas = pickle.load(f)
        with open(ids_path, 'rb') as f:
            ids = pickle.load(f)
        with open(documents_path, 'rb') as f:
            documents_text = pickle.load(f)
            
        # 返回包含索引和相关数据的对象
        return {
            'index': index,
            'metadatas': metadatas,
            'ids': ids,
            'documents': documents_text
        }
    
    print(f"Creating new FAISS summary collection: {collection_name}")
    
    # 确保保存目录存在
    os.makedirs(persist_directory, exist_ok=True)
    
    # 生成摘要和准备数据
    ids = []
    documents_text = []
    metadatas = []
    for i, page in enumerate(tqdm(pages, desc="Generating summaries")):
        summary = generate_summary(page["text"])
        ids.append(f"summary_{i}")
        documents_text.append(summary)
        metadatas.append(page["metadata"])
    
    # 创建嵌入向量
    embeddings = embedding_model.embed_documents(documents_text)
    embedding_dim = len(embeddings[0])
    
    # 初始化FAISS索引
    index = faiss.IndexFlatL2(embedding_dim)
    
    # 添加向量到索引
    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)
    
    # 保存索引和相关数据
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadatas, f)
    with open(ids_path, 'wb') as f:
        pickle.dump(ids, f)
    with open(documents_path, 'wb') as f:
        pickle.dump(documents_text, f)
    
    print(f"Added {len(pages)} summaries to FAISS summary index")
    
    # 返回包含索引和相关数据的对象
    return {
        'index': index,
        'metadatas': metadatas,
        'ids': ids,
        'documents': documents_text
    }

def summary_search(collection, query, k=5):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    
    # 将查询向量转换为FAISS需要的格式
    query_vector = np.array([query_embedding]).astype('float32')
    
    # 使用FAISS进行相似性搜索
    distances, indices = collection['index'].search(query_vector, k)
    
    # 处理搜索结果
    output = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        output.append({
            "text": collection['documents'][idx],
            "metadata": collection['metadatas'][idx],
            "similarity": 1 - distances[0][i]  # 将L2距离转换为相似度分数
        })
    
    return output