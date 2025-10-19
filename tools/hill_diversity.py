import pandas as pd
import re

# 设置文件路径
input_path = r"F:\\destop\\video_sample\\diatribution_of_pedestrian.xlsx"
output_path = r"F:\\destop\\video_sample\\hill_diversity.xlsx"

# 读取 Excel 文件
df = pd.read_excel(input_path)

# 定义 6 种活动关键词
activity_keywords = {
    "Walking": r"行走",
    "Staying": r"停留",
    "Talking": r"交流",
    "Phone": r"玩手机",
    "Calling": r"打电话",
    "Dogwalking": r"遛狗"
}

# 定义解析函数
def parse_activities(detail_text):
    result = {}
    if pd.isna(detail_text):
        for key in activity_keywords:
            result[key] = 0
        return pd.Series(result)

    for key, pattern in activity_keywords.items():
        result[key] = len(re.findall(pattern, str(detail_text)))
    return pd.Series(result)

# 应用函数批量处理所有行
activity_counts = df["details"].apply(parse_activities)

# 合并结果并输出
df_merged = pd.concat([df, activity_counts], axis=1)
df_merged.to_excel(output_path, index=False)
print("文件已成功保存到：", output_path)
