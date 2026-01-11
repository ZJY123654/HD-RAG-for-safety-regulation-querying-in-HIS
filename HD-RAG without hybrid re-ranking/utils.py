import os
from glob import glob
from tqdm import tqdm
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import re
from openai import OpenAI

# OpenAI client for summaries
client = OpenAI(
    base_url="your base url",
    api_key="your api key"
)

def get_embedding_model():
    api_key = "your api key"
    if not api_key:
        raise ValueError("API密钥未设置")
    openai_api_base = "your base url"
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=openai_api_base
    )
    return embedding_model

def generate_summary(text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Summarize the following text concisely."},
            {"role": "user", "content": text}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_pages_from_pdf(pdf_path):
    print(f"Extracting pages from {pdf_path}...")
    pdf_reader = PdfReader(pdf_path)
    pages = []
    for page_num in tqdm(range(len(pdf_reader.pages)), desc=f"Processing pages of {os.path.basename(pdf_path)}"):
        text = pdf_reader.pages[page_num].extract_text()
        if len(text.strip()) > 50:
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_num + 1
                }
            })
    print(f"Extracted {len(pages)} pages with content")
    return pages

def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks_text = text_splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        if len(chunk_text.strip()) > 50:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "is_summary": False
            })
            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))
    return chunks

def process_multiple_documents(data_dir):
    all_pages = []
    pdf_files = []
    for file_pattern in ["**/*.pdf"]:
        pdf_files.extend(glob(os.path.join(data_dir, file_pattern), recursive=True))
    
    for file_path in tqdm(pdf_files, desc="Processing documents"):
        try:
            pages = extract_pages_from_pdf(file_path)
            all_pages.extend(pages)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    print(f"Processed {len(all_pages)} pages from {data_dir}")
    return all_pages