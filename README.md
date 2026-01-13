# HD-RAG-for-safety-regulation-querying-in-HIS

## Project Overview

HD-RAG is a hierarchical hybrid Retrieval-Augmented Generation (RAG) system specifically designed for the hydropower and water resources engineering domain. It aims to improve the accuracy and efficiency of querying safety regulation documents. By leveraging a hierarchical indexing structure and a hybrid retrieval mechanism, the system enables efficient retrieval and precise answering over complex and lengthy documents such as safety regulations in hydropower engineering.

## Project Structure

The project includes five main implementation variants for comparing the performance of different retrieval strategies:

```
HD-RAG/
├── HD-RAG/                     # Full hierarchical hybrid RAG system
├── HD-RAG without hierarchical index/  # RAG system without hierarchical indexing
├── HD-RAG without hybrid re-ranking/   # RAG system without hybrid re-ranking
├── Keyword-only RAG/           # Keyword-based RAG system only
├── Vector-only RAG/            # Vector-based RAG system only
├── data/                       # Dataset directory
└── requirements.txt            # Project dependencies
```

### Core Module Description

#### HD-RAG（Full System）
- `main.py`: Main program entry point that coordinates index construction, retrieval, and response generation
- `hierarchical_retrieval.py`: Implementation of the hierarchical retrieval algorithm
- `summary_index.py`: Construction and management of the document summary index
- `detail_index.py`: Construction and management of the document detail index
- `response_generation.py`: Answer generation based on retrieved results
- `utils.py`: Utility functions

#### Evaluation Tools
- `bacth_evaluation.ipynb`: Batch evaluation script
- `calculate_averages.py`: Computation of average evaluation metrics
- `RAGAS.ipynb`: Integration of the RAGAS evaluation framework

## Technology Stack

- Python 3.10
- FAISS (vector indexing)
- BM25 (keyword retrieval)
- LangChain (RAG framework)
- OpenAI (generation models)
- Pandas (data processing)
- NumPy (numerical computation)

## Installation

1. Clone the project to your local machine

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare the data
Place the PDF documents in the data/ directory

## Usage

### 1. Index Construction

Run the following in the HD-RAG/ directory:

```python
# Build indices (default uses the data directory)
summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index = build_indices(force_rebuild=False)
```

Set`force_rebuild=True`to forcibly rebuild all indices.

### 2. Retrieval and Generation

1. First, modify the query CSV file path in `main.py`：
   ```python
   # Modify this line to your CSV file path
   csv_path = r"your csv path"
   ```

2. Then run：
   ```bash
   python main.py
   ```

The CSV file should contain a column named "Query", with each row representing a query question.

### 3. System Evaluation

Use the batch evaluation script:

```bash
jupyter notebook bacth_evaluation.ipynb
```

## System Architecture
Hierarchical Index Structure

1.Summary Index Layer: A vector index constructed from document summaries, used to quickly locate relevant document sections
2.Detail Index Layer: A vector index and BM25 index constructed from full document text, used for fine-grained retrieval of relevant information

### Hybrid Retrieval Workflow

1.Candidate documents are first selected using the summary index
2.Detailed retrieval is then performed within the candidate document scope
3.A hybrid re-ranking algorithm is applied (the parameter α controls the relative weights of vector-based and keyword-based retrieval)
4.Final answers are generated based on the retrieved results

## Comparison

| Method | Hierarchical Index | Hybrid Re-ranking | Retrieval Strategy |
|------|----------|------------|----------|
| HD-RAG | ✅ | ✅ | Hierarchical hybrid retrieval |
| HD-RAG without hierarchical index | ❌ | ✅ | Single-layer hybrid retrieval |
| HD-RAG without hybrid re-ranking | ✅ | ❌ | Hierarchical single-modality retrieval |
| Keyword-only RAG | ❌ | ❌ | Keyword-only retrieval |
| Vector-only RAG | ❌ | ❌ | Vector-only retrieval |

## Evaluation Metrics

The system is evaluated using the following metrics:
- Context Precision
- Context Relevancy
- Answer Accuracy
- Answer Relevancy
- Answer Faithfulness

## Configuration Parameters

Key configuration parameters include:

k_summary: Number of documents retrieved from the summary index (default: 5)

k_detail: Number of documents retrieved from the detail index (default: 5)

alpha: Weight between vector-based and keyword-based retrieval (default: 0.8)

## Dataset

The project uses PDF documents related to hydropower and water resources engineering, including:

1.Technical Specifications for Construction Safety of Hydropower and Water Resources Projects
2.Guidelines for Hazard Identification and Risk Assessment in Hydropower and Water Resources Engineering
3.Specifications for Construction Organization Design of Hydropower and Water Resources Projects


