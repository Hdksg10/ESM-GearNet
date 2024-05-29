import pandas as pd

# 读取 CSV 文件
df1 = pd.read_csv('./ac.csv')
df2 = pd.read_csv('./sel.csv')

# 处理第一个 CSV 文件
df1.columns = ['mutation', 'activity'] + list(df1.columns[2:])  # 重命名第二列为 'activity'

# 处理第二个 CSV 文件
df2 = df2.drop(df2.columns[0], axis=1)  # 删除第一列
df2.columns = ['selectivity'] + list(df2.columns[1:])  # 重命名第二列为 'selectivity'

# 合并两个 DataFrame
merged_df = pd.concat([df1, df2], axis=1)

# 保存合并后的结果到新文件
merged_df.to_csv('submission.csv', index=False)