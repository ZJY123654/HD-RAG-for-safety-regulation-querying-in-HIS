import faiss
import numpy as np
import os
import pickle
from langchain.schema import Document
from utils import get_embedding_model
from tqdm import tqdm

# 创建一个包装类来保持与原有接口的一致性
class FAISSVectorStore:
    def __init__(self, index, documents, metadatas):
        self.index = index
        self.documents = documents  # 存储文本内容
        self.metadatas = metadatas  # 存储元数据

    def query(self, query_embeddings, n_results):
        # FAISS查询返回距离和索引
        distances, indices = self.index.search(query_embeddings, n_results)
        
        # 构建与原接口兼容的结果格式
        results = {
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        for i in range(len(query_embeddings)):
            docs = [self.documents[idx] for idx in indices[i] if idx < len(self.documents)]
            metas = [self.metadatas[idx] for idx in indices[i] if idx < len(self.metadatas)]
            dists = distances[i][:len(docs)].tolist()
            
            results["documents"].append(docs)
            results["metadatas"].append(metas)
            results["distances"].append(dists)
        
        return results

    def get(self):
        # 返回所有文档
        return {
            "documents": self.documents,
            "metadatas": self.metadatas
        }

def load_or_create_vector_store(documents, persist_directory="./faiss_db", collection_name="fusion_rag"):
    # 创建持久化目录
    os.makedirs(persist_directory, exist_ok=True)
    
    # 定义文件路径
    index_path = os.path.join(persist_directory, f"{collection_name}_index.faiss")
    docs_path = os.path.join(persist_directory, f"{collection_name}_docs.pkl")
    metas_path = os.path.join(persist_directory, f"{collection_name}_metas.pkl")
    
    # 检查是否存在已保存的索引
    if os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(metas_path):
        print(f"Loading existing FAISS index: {collection_name}")
        
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        
        # 加载文档和元数据
        with open(docs_path, 'rb') as f:
            documents_text = pickle.load(f)
        with open(metas_path, 'rb') as f:
            metadatas = pickle.load(f)
        
        # 创建FAISSVectorStore实例
        vector_store = FAISSVectorStore(index, documents_text, metadatas)
    else:
        print(f"Creating new FAISS index: {collection_name}")
        
        # 提取文档内容和元数据
        documents_text = []
        metadatas = []
        for doc in tqdm(documents, desc="处理文档", unit="个"):
            documents_text.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        # 获取嵌入模型并生成嵌入向量
        embedding_model = get_embedding_model()
        print("正在生成文档嵌入向量...")
        embeddings = embedding_model.embed_documents(documents_text)
        print("文档嵌入向量生成完成")
        
        # 将嵌入向量转换为numpy数组
        embeddings_np = np.array(embeddings).astype('float32')
        
        # 获取嵌入向量的维度
        dimension = embeddings_np.shape[1]
        
        # 创建FAISS索引
        index = faiss.IndexFlatL2(dimension)  # 使用L2距离
        index.add(embeddings_np)  # 添加向量到索引
        
        # 保存索引和文档
        faiss.write_index(index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(documents_text, f)
        with open(metas_path, 'wb') as f:
            pickle.dump(metadatas, f)
        
        # 创建FAISSVectorStore实例
        vector_store = FAISSVectorStore(index, documents_text, metadatas)
        
        print(f"Added {len(documents)} items to FAISS vector store")
    
    return vector_store

def similarity_search_with_scores(collection, query, k=5):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)
    # 将查询嵌入转换为2D数组以匹配FAISS的输入要求
    query_embedding_2d = np.array([query_embedding]).astype('float32')
    results = collection.query(query_embedding_2d, k)
    
    output = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i]  # 转换为相似度分数
            })
    
    return output

def get_all_documents(collection):
    results = collection.get()
    return [{"text": doc, "metadata": meta} for doc, meta in zip(results["documents"], results["metadatas"]) if doc]