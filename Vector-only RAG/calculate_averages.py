import pandas as pd
import numpy as np

# 读取Excel文件
excel_file = 'your excel file path'
df = pd.read_excel(excel_file)

# 定义需要计算平均值的列
columns_to_average = [
    'Context_Precision', 
    'Context_Relevance', 
    'Answer_Accuracy', 
    'Semantic_Similarity', 
    'Faithfulness', 
    'Evaluation_Time'
]

# 定义行范围
ranges = [
    (1, 100),   # 第1-100条
    (101, 213), # 第101-213条
    (214, 321), # 第214-321条
    (322, 420)  # 第322-420条
]

# 计算每个范围内各列的平均值
results = []
for start, end in ranges:
    # 注意：pandas的索引是从0开始的，所以需要减1
    range_df = df.iloc[start-1:end]
    range_avg = range_df[columns_to_average].mean()
    results.append({
        'range': f'第{start}-{end}条',
        **range_avg.to_dict()
    })

# 计算所有420条的汇总平均值
total_avg = df.iloc[:420][columns_to_average].mean()
results.append({
    'range': '所有420条汇总',
    **total_avg.to_dict()
})

# 将结果保存到txt文件
output_file = 'evaluation_averages.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("各范围评估指标平均值如下：\n\n")
    for result in results:
        f.write(f"{result['range']}\n")
        for column in columns_to_average:
            f.write(f"{column}: {result[column]:.4f}\n")
        f.write("\n")

print(f"计算完成，结果已保存到 {output_file}")