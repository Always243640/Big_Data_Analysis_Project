import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))


def _resolve_path(*parts):
    return os.path.join(BASE_DIR, *parts)


# 读取原始数据
file_path = _resolve_path('total.csv')
df = pd.read_csv(file_path)

# 步骤 1: 处理日期格式，填充缺失的空格
# 先将续借时间和还书时间的格式进行处理
df['续借时间'] = df['续借时间'].astype(str).apply(lambda x: x.replace('DD', 'DD ').replace('HH', 'HH '))
df['还书时间'] = df['还书时间'].astype(str).apply(lambda x: x.replace('DD', 'DD ').replace('HH', 'HH '))

# 步骤 2: 解析日期并处理 user_id 与 book_id
df['借阅时间'] = pd.to_datetime(df['借阅时间'], errors='coerce')
df['续借时间'] = pd.to_datetime(df['续借时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['还书时间'] = pd.to_datetime(df['还书时间'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce').astype('Int64')
df['book_id'] = pd.to_numeric(df['book_id'], errors='coerce').astype('Int64')

# 创建新的时间列，如果续借时间和借阅时间都存在则取最新时间
df['timestamp'] = df[['借阅时间', '续借时间']].max(axis=1)

# 步骤 3: 构建新的数据框
new_df = df[['user_id', 'book_id', 'timestamp']].dropna(subset=['user_id', 'book_id', 'timestamp'])

# 步骤 4: 排序数据
# 先按user_id升序排序，再按timestamp升序排序
new_df = new_df.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
new_df['timestamp'] = new_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 步骤 5: 关联用户年级与图书二级分类信息
user_file_path = os.path.join(PROJECT_ROOT, 'mydata', 'user.csv')
item_file_path = os.path.join(PROJECT_ROOT, 'mydata', 'item.csv')

user_df = pd.read_csv(user_file_path)[['借阅人', '年级']].rename(columns={'借阅人': 'user_id'})
item_df = pd.read_csv(item_file_path)[['book_id', '二级分类']]

new_df = new_df.merge(user_df, on='user_id', how='left')
new_df = new_df.merge(item_df, on='book_id', how='left')
new_df = new_df[['user_id', 'book_id', 'timestamp', '年级', '二级分类']]

# 保存结果到新的CSV文件
new_file_path = _resolve_path('interaction.csv')
new_df.to_csv(new_file_path, index=False)

print(f"处理完成，结果已保存到 {new_file_path}")
