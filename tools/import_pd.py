import pandas as pd
import re

# 你的输入和输出路径
input_path = r"F:/destop/video_sample/diatribution.csv"
output_path = r"F:/destop/video_sample/diatribution_of_grids.csv"

# 读取 CSV 文件
try:
    df = pd.read_csv(input_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding='gbk')

# 处理每一行的 details 字段
def final_parse_details(detail_text):
    if pd.isna(detail_text):
        return pd.Series([0, 0, 0, 0])  # 成人、儿童、老人、活动类型数

    # 匹配人群类型
    people = re.findall(r'\d+号(.*?)：', detail_text)
    adult_count = people.count('成人')
    child_count = people.count('儿童')
    elderly_count = people.count('老人')

    # 主活动 + 其他活动
    main_activities = re.findall(r'：(.*?)。其他活动', detail_text)
    other_activities = re.findall(r'其他活动：(.*?)[,，]', detail_text + ',')  # 结尾补一个逗号方便匹配

    all_activities = set()
    for a in main_activities + other_activities:
        a = a.strip()
        if a and a != '无':
            all_activities.add(a)

    return pd.Series([adult_count, child_count, elderly_count, len(all_activities)])

# 应用处理函数
df[['adult_count', 'child_count', 'elderly_count', 'activity_type_count']] = df['details'].apply(final_parse_details)

# 保存为新 CSV
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("🎉 成功导出！路径为：", output_path)

