import os
import time
import pandas as pd
from hierarchical_retrieval import hierarchical_retrieval
from response_generation import generate_response
from summary_index import load_or_create_summary_index
from detail_index import load_or_create_detail_index
from utils import process_multiple_documents

"""
Qualitative Analysis 示例

本脚本用于演示HiKRAG系统从检索到生成答案的完整过程，包括：
1. 索引加载
2. 层次化检索（摘要级和详细级）
3. 检索结果分析
4. 答案生成

通过一个具体的查询示例，展示系统的工作流程和中间结果。
"""

def load_indices(data_dir="data", force_rebuild=False):
    """加载已构建的索引
    
    根据用户要求，默认情况下直接加载已构建好的索引，不进行重建。
    只有在明确指定force_rebuild=True时才会重建索引。
    """
    print("===== 加载已构建的索引 =====")
    print("注意: 根据用户要求，默认直接加载已有的索引文件，不进行重建")
    
    # 检查索引文件是否存在
    required_files = [
        "./faiss_summary_index.index", 
        "./faiss_summary_data.pkl", 
        "./faiss_detail_index.index", 
        "./faiss_detail_data.pkl", 
        "./bm25_index.pkl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"警告: 缺少以下索引文件: {', '.join(missing_files)}")
        if force_rebuild:
            print("将重新构建索引...")
        else:
            print("正在使用现有文件并构建缺失的索引...")
    else:
        print("所有索引文件都存在，可以直接加载")
    
    # 只在需要时处理文档
    pages = None
    if force_rebuild or missing_files:
        print("\n处理文档以创建/更新索引...")
        pages = process_multiple_documents(data_dir)
    
    # 加载摘要索引
    print("\n加载摘要索引...")
    start_time = time.time()
    try:
        if pages:
            summary_index, summary_texts, summary_metas = load_or_create_summary_index(pages)
        else:
            # 尝试直接加载已有的索引
            summary_index, summary_texts, summary_metas = load_or_create_summary_index()
    except Exception as e:
        print(f"加载摘要索引时出错: {e}")
        print("尝试处理文档后再加载...")
        if not pages:
            pages = process_multiple_documents(data_dir)
        summary_index, summary_texts, summary_metas = load_or_create_summary_index(pages)
    
    summary_time = time.time() - start_time
    print(f"摘要索引加载完成，耗时: {summary_time:.2f}秒")
    print(f"摘要文档数量: {len(summary_texts)}")
    
    # 加载详细索引和BM25索引
    print("\n加载详细索引和BM25索引...")
    start_time = time.time()
    try:
        if pages:
            detail_index, detail_texts, detail_metas, bm25_index = load_or_create_detail_index(pages)
        else:
            # 尝试直接加载已有的索引
            detail_index, detail_texts, detail_metas, bm25_index = load_or_create_detail_index()
    except Exception as e:
        print(f"加载详细索引时出错: {e}")
        print("尝试处理文档后再加载...")
        if not pages:
            pages = process_multiple_documents(data_dir)
        detail_index, detail_texts, detail_metas, bm25_index = load_or_create_detail_index(pages)
    
    detail_time = time.time() - start_time
    print(f"详细索引加载完成，耗时: {detail_time:.2f}秒")
    print(f"详细文档数量: {len(detail_texts)}")
    
    return summary_index, summary_texts, summary_metas, detail_index, detail_texts, detail_metas, bm25_index

def analyze_retrieval_process(query, summary_index, summary_texts, summary_metas, 
                            detail_index, detail_texts, detail_metas, bm25_index,
                            top_k_summary=3, top_k_detail=3, alpha=0.8):
    """分析层次化检索过程"""
    print(f"\n===== 分析检索过程 - 查询: '{query}' =====")
    
    # 保存中间结果以便分析
    analysis_results = {
        "query": query,
        "summary_retrieval": [],
        "detail_retrieval": [],
        "final_ranking": []
    }
    
    # 第一阶段：摘要级检索
    print("\n1. 第一阶段：摘要级向量检索")
    print(f"   检索前{top_k_summary}个相关页面摘要")
    
    # 导入summary_search以直接调用并查看结果
    from summary_index import summary_search
    relevant_summaries = summary_search(summary_index, summary_texts, summary_metas, query, k=top_k_summary)
    
    for i, summary in enumerate(relevant_summaries, 1):
        print(f"\n   摘要 {i}:")
        print(f"   相似度分数: {summary['similarity']:.4f}")
        print(f"   来源: {summary['metadata']['source']}")
        print(f"   页码: {summary['metadata']['page']}")
        print(f"   内容: {summary['text'][:100]}...")
        
        analysis_results["summary_retrieval"].append({
            "rank": i,
            "score": summary['similarity'],
            "source": summary['metadata']['source'],
            "page": summary['metadata']['page'],
            "text": summary['text']
        })
    
    # 第二阶段：详细级混合检索
    print("\n2. 第二阶段：详细级混合检索（向量+BM25）")
    print(f"   在筛选的页面内进行混合检索，权重系数 alpha={alpha}")
    
    # 使用我们的层次化检索函数
    retrieved_docs = hierarchical_retrieval(
        query, summary_index, summary_texts, summary_metas, 
        detail_index, detail_texts, detail_metas, bm25_index,
        top_k_summary=top_k_summary, top_k_detail=top_k_detail, alpha=alpha
    )
    
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n   文档 {i}:")
        print(f"   组合分数: {doc['combined_score']:.4f} (向量分数: {doc['vector_score']:.4f}, BM25分数: {doc['bm25_score']:.4f})")
        print(f"   来源: {doc['metadata']['source']}")
        print(f"   页码: {doc['metadata']['page']}")
        print(f"   内容片段: {doc['text'][:150]}...")
        
        analysis_results["final_ranking"].append({
            "rank": i,
            "combined_score": doc['combined_score'],
            "vector_score": doc['vector_score'],
            "bm25_score": doc['bm25_score'],
            "source": doc['metadata']['source'],
            "page": doc['metadata']['page'],
            "text": doc['text']
        })
    
    return retrieved_docs, analysis_results

def generate_and_analyze_response(query, retrieved_docs):
    """生成答案并分析"""
    print("\n===== 生成答案 =====")
    
    # 构建上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
    print(f"构建的上下文长度: {len(context)} 字符")
    
    # 生成答案
    start_time = time.time()
    response = generate_response(query, context)
    generation_time = time.time() - start_time
    
    print(f"\n生成的答案 (耗时: {generation_time:.2f}秒):")
    print(response)
    
    # 分析上下文和答案的关系
    print("\n===== 关系分析 =====")
    relevant_docs_count = 0
    for i, doc in enumerate(retrieved_docs, 1):
        # 简单的相关性判断（检查答案是否引用了文档中的关键内容）
        # 这里使用简单的关键词匹配，实际应用中可以使用更复杂的方法
        is_relevant = any(keyword in response.lower() for keyword in doc["text"].lower().split()[:20])
        if is_relevant:
            relevant_docs_count += 1
            print(f"文档 {i}: 相关")
        else:
            print(f"文档 {i}: 可能不相关")
    
    print(f"\n相关性统计: {relevant_docs_count}/{len(retrieved_docs)} 个检索到的文档在答案中被引用")
    
    return response

def run_qualitative_analysis(sample_queries=None, output_file="qualitative_analysis_results.csv", force_rebuild=False):
    """运行定性分析"""
    # 默认查询样例
    if sample_queries is None:
        sample_queries = [
            "机组启动试运行包含哪些主要阶段？"
        ]
    
    # 加载索引
    indices = load_indices(force_rebuild=force_rebuild)
    
    # 保存所有分析结果
    all_results = []
    
    for query in sample_queries:
        print("\n" + "="*80)
        print(f"处理查询: {query}")
        print("="*80)
        
        # 分析检索过程
        retrieved_docs, retrieval_analysis = analyze_retrieval_process(query, *indices)
        
        # 生成并分析答案
        response = generate_and_analyze_response(query, retrieved_docs)
        
        # 保存结果
        all_results.append({
            "query": query,
            "retrieved_docs_count": len(retrieved_docs),
            "top_summary_score": retrieval_analysis["summary_retrieval"][0]["score"] if retrieval_analysis["summary_retrieval"] else None,
            "top_doc_score": retrieval_analysis["final_ranking"][0]["combined_score"] if retrieval_analysis["final_ranking"] else None,
            "response": response
        })
    
    # 保存到CSV
    if output_file:
        pd.DataFrame(all_results).to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: {output_file}")
    
    print("\n===== 定性分析完成 =====")
    return all_results

def example_usage():
    """
    定性分析示例使用方法
    
    这个函数展示了如何以不同方式使用定性分析脚本：
    1. 使用默认查询进行分析（默认不重建索引）
    2. 提供自定义查询列表
    3. 强制重建索引并进行分析（仅在需要时使用）
    4. 保存结果到自定义文件
    """
    print("\n===== HiKRAG 定性分析工具使用示例 =====\n")
    print("重要提示: 脚本默认直接加载已构建的索引，不进行重建操作")
    
    # 示例1: 使用默认查询和默认设置（推荐方式）
    print("\n示例1: 使用默认查询（直接加载现有索引）")
    # run_qualitative_analysis()  # 默认不重建索引
    
    # 示例2: 使用自定义查询列表
    print("\n示例2: 使用自定义查询")
    custom_queries = [
        "水利水电工程施工中如何预防高处坠落事故？",
        "电气设备安装的安全技术规程是什么？",
        "重大事故隐患判定的标准有哪些？"
    ]
    # run_qualitative_analysis(sample_queries=custom_queries)  # 默认不重建索引
    
    # 示例3: 强制重建索引并分析（仅在必要时使用）
    print("\n示例3: 强制重建索引（仅在数据更新或索引损坏时使用）")
    print("警告: 此操作会删除现有索引并重新构建，耗时较长")
    # run_qualitative_analysis(force_rebuild=True)
    
    # 示例4: 保存结果到自定义文件
    print("\n示例4: 保存到自定义文件")
    # run_qualitative_analysis(output_file="custom_analysis_results.csv")
    
    print("\n要运行这些示例，请取消相应行的注释")
    print("或直接运行: python qualitative_analysis.py\n")

if __name__ == "__main__":
    # 显示使用示例
    example_usage()
    
    # 默认运行 - 使用默认查询进行定性分析，不重建索引
    print("开始默认定性分析（直接加载现有索引）...")
    run_qualitative_analysis(force_rebuild=False)