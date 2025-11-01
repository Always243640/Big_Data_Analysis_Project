import pandas as pd
from datetime import datetime

# 读取原始数据
file_path = r'total.csv'
df = pd.read_csv(file_path)

# 步骤 1: 处理日期格式，填充缺失的空格
# 先将续借时间和还书时间的格式进行处理
df['续借时间'] = df['续借时间'].astype(str).apply(lambda x: x.replace('DD', 'DD ').replace('HH', 'HH '))
df['还书时间'] = df['还书时间'].astype(str).apply(lambda x: x.replace('DD', 'DD ').replace('HH', 'HH '))

# 步骤 2: 提取 user_id, book_id 和 time
# 对于每一条交互记录，如果有续借时间，将续借时间作为时间戳，如果没有，则取借阅时间
df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
df['续借时间'] = pd.to_datetime(df['续借时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['还书时间'] = pd.to_datetime(df['还书时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


# 创建新的time列，选择续借时间或借阅时间，取续借时间优先
df['time'] = df.apply(lambda row: row['续借时间'] if pd.notnull(row['续借时间']) else row['借阅时间'], axis=1)

# 步骤 3: 提取 user_id, book_id, time
new_df = df[['user_id', 'book_id', 'time']]

# 步骤 4: 排序数据
# 先按user_id升序排序，再按time升序排序
new_df = new_df.sort_values(by=['user_id', 'time'], ascending=[True, True])

# 步骤 5: 删除time列，只保留user_id, book_id
new_df = new_df[['user_id', 'book_id']]

# 保存结果到新的CSV文件
new_file_path = r'interaction.csv'
new_df.to_csv(new_file_path, index=False)

print(f"处理完成，结果已保存到 {new_file_path}")
