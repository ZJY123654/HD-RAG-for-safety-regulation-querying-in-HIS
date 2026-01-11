import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm
from utils import get_embedding_model, generate_summary

def load_or_create_summary_index(pages, index_path="./faiss_summary_index.index", data_path="./faiss_summary_data.pkl"):
    if os.path.exists(index_path) and os.path.exists(data_path):
        index = faiss.read_index(index_path)
        with open(data_path, 'rb') as f:
            documents_text, metadatas = pickle.load(f)
        print(f"Loaded existing Faiss summary index")
    else:
        embedding_model = get_embedding_model()
        documents_text = []
        metadatas = []
        for page in tqdm(pages, desc="生成摘要", unit="页"):
            summary = generate_summary(page["text"])
            documents_text.append(summary)
            metadatas.append(page["metadata"])
        
        embeddings = embedding_model.embed_documents(documents_text)
        emb_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(emb_array)
        
        dimension = emb_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(emb_array)
        
        faiss.write_index(index, index_path)
        with open(data_path, 'wb') as f:
            pickle.dump((documents_text, metadatas), f)
        print(f"Created and saved Faiss summary index with {len(pages)} summaries")
    
    return index, documents_text, metadatas

def summary_search(index, documents_text, metadatas, query, k=5):
    embedding_model = get_embedding_model()
    query_embedding = np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    output = []
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx != -1:
            output.append({
                "text": documents_text[idx],
                "metadata": metadatas[idx],
                "similarity": D[0][i]
            })
    return output