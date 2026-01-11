import csv
import re

# 读取CSV文件并计算平均token长度
def calculate_average_token_length(csv_file_path):
    # 初始化计数器
    total_context_tokens = 0
    total_response_tokens = 0
    row_count = 0
    
    try:
        # 读取文件内容并过滤掉NUL字符
        with open(csv_file_path, 'r', encoding='utf-8', errors='replace') as csvfile:
            content = csvfile.read().replace('\x00', '')
        
        # 使用StringIO创建内存文件对象
        import io
        csvfile = io.StringIO(content)
        
        # 使用csv.reader读取文件
        csv_reader = csv.DictReader(csvfile)
            
        for row in csv_reader:
                # 获取Retrievaled_context和Response列的值
                context = row.get('Retrievaled_context', '').strip()
                response = row.get('Response', '').strip()
                
                # 计算token数量（这里简单地以字符数作为token数的近似值）
                # 对于中文，每个汉字可以近似为一个token
                context_tokens = len(context)
                response_tokens = len(response)
                
                # 累加token数量
                total_context_tokens += context_tokens
                total_response_tokens += response_tokens
                row_count += 1
        
        # 计算平均值
        if row_count > 0:
            avg_context_tokens = total_context_tokens / row_count
            avg_response_tokens = total_response_tokens / row_count
            
            return {
                'total_rows': row_count,
                'average_context_tokens': avg_context_tokens,
                'average_response_tokens': avg_response_tokens
            }
        else:
            return {
                'total_rows': 0,
                'average_context_tokens': 0,
                'average_response_tokens': 0
            }
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

# 主函数
def main():
    csv_file_path = 'rag_results.csv'
    
    print(f"正在计算文件 {csv_file_path} 中Retrievaled_context和Response的平均token长度...")
    result = calculate_average_token_length(csv_file_path)
    
    if result:
        print(f"总行数: {result['total_rows']}")
        print(f"Retrievaled_context的平均token长度: {result['average_context_tokens']:.2f}")
        print(f"Response的平均token长度: {result['average_response_tokens']:.2f}")
    else:
        print("计算失败。")

if __name__ == "__main__":
    main()