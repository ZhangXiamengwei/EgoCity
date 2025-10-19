import pandas as pd
import numpy as np
import re

# 路径设置
input_path = r"F:\\destop\\video_sample\\hill_diversity.xlsx"
output_path = r"F:\\destop\\video_sample\\hill_utilization_result.xlsx"

# 读取数据
df = pd.read_excel(input_path)

# 定义活动列（6种）
activity_columns = ["Walking", "Staying", "Talking", "Phone", "Calling", "Dogwalking"]

# 定义计算函数
def calculate_hill_diversity(row):
    total = row.get("person_count", 0)
    if total == 0:
        return pd.Series({"H_shannon": 0, "Hill_number": 0, "Utilization": 0})

    # 计算每个活动的占比 p_i
    p_list = [row[col] / total for col in activity_columns if row[col] > 0]
    
    # 计算 Shannon H'
    H = -sum(p * np.log(p) for p in p_list)
    
    # 计算 Hill diversity
    D = np.exp(H)

    # 最终利用率
    utilization = total * D

    return pd.Series({
        "H_shannon": H,
        "Hill_number": D,
        "Utilization": utilization
    })

# 应用函数
metrics = df.apply(calculate_hill_diversity, axis=1)

# 合并保存
df_final = pd.concat([df, metrics], axis=1)
df_final.to_excel(output_path, index=False)

print("✅ 修正后：利用率计算完成，已保存到：", output_path)

