# HD-RAG-for-safety-regulation-querying-in-HIS

## 项目简介

HD-RAG是一个专为水利水电工程领域设计的分层混合检索增强生成系统，旨在提高安全规程文档的查询准确性和效率。该系统通过分层索引结构和混合检索机制，实现了对水利水电工程安全规程等复杂长文档的高效检索和精确回答。

## 项目结构

项目包含五个主要实现版本，用于比较不同检索策略的性能：

```
HD-RAG/
├── HD-RAG/                     # 完整的分层混合RAG系统
├── HD-RAG without hierarchical index/  # 无分层索引的RAG系统
├── HD-RAG without hybrid re-ranking/   # 无混合重排序的RAG系统
├── Keyword-only RAG/           # 仅基于关键词的RAG系统
├── Vector-only RAG/            # 仅基于向量的RAG系统
├── data/                       # 数据集目录
└── requirements.txt            # 项目依赖
```

### 核心模块说明

#### HD-RAG（完整系统）
- `main.py`: 主程序入口，协调索引构建、检索和生成流程
- `hierarchical_retrieval.py`: 分层检索算法实现
- `summary_index.py`: 文档摘要索引构建与管理
- `detail_index.py`: 文档细节索引构建与管理
- `response_generation.py`: 基于检索结果生成回答
- `utils.py`: 工具函数集

#### 评估工具
- `bacth_evaluation.ipynb`: 批量评估脚本
- `calculate_averages.py`: 计算评估指标平均值
- `RAGAS.ipynb`: RAGAS评估框架集成

## 技术栈

- Python 3.10
- FAISS (向量索引)
- BM25 (关键词检索)
- LangChain (RAG框架)
- OpenAI (生成模型)
- Pandas (数据处理)
- NumPy (数值计算)

## 安装说明

1. 克隆项目到本地

2. 安装依赖包
```bash
pip install -r requirements.txt
```

3. 准备数据
将PDF文档放入`data/`目录下

## 使用方法

### 1. 构建索引

在`HD-RAG/`目录下运行：

```python
# 构建索引（默认使用data目录）
summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index = build_indices(force_rebuild=False)
```

设置`force_rebuild=True`可以强制重建所有索引。

### 2. 运行检索与生成

1. 首先修改`main.py`中的查询CSV文件路径：
   ```python
   # 将此行修改为您的CSV文件路径
   csv_path = r"your csv path"
   ```

2. 然后运行：
   ```bash
   python main.py
   ```

CSV文件应包含"Query"列，每行为一个查询问题。

### 3. 评估系统性能

使用批量评估脚本：

```bash
jupyter notebook bacth_evaluation.ipynb
```

## 系统架构

### 分层索引结构
1. **摘要索引层**：使用文档摘要构建的向量索引，用于快速定位相关文档段落
2. **细节索引层**：使用文档全文构建的向量索引和BM25索引，用于精确检索相关信息

### 混合检索流程
1. 首先通过摘要索引快速筛选出候选文档
2. 然后在候选文档范围内进行细节检索
3. 使用混合重排序算法（α参数控制向量和关键词权重）
4. 基于检索结果生成最终回答

## 各版本比较

| 版本 | 分层索引 | 混合重排序 | 检索策略 |
|------|----------|------------|----------|
| HD-RAG | ✅ | ✅ | 分层混合检索 |
| HD-RAG without hierarchical index | ❌ | ✅ | 单层混合检索 |
| HD-RAG without hybrid re-ranking | ✅ | ❌ | 分层单模态检索 |
| Keyword-only RAG | ❌ | ❌ | 仅关键词检索 |
| Vector-only RAG | ❌ | ❌ | 仅向量检索 |

## 评估指标

系统使用以下指标进行评估：
- Context Precision
- Context Relevancy
- Answer Accuracy
- Answer Relevancy
- Answer Faithfulness

## 配置参数

主要配置参数包括：
- `k_summary`: 摘要索引检索的文档数量（默认：5）
- `k_detail`: 细节索引检索的文档数量（默认：5）
- `alpha`: 向量检索与关键词检索的权重（默认：0.8）

## 数据集

项目使用水利水电工程相关的PDF文档作为数据集，包含：
- 水利水电工程施工安全技术规程
- 水利水电工程危险源辨识与风险评价导则
- 水利水电工程施工组织设计规范等

