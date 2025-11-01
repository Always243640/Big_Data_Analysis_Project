import pandas as pd

# 读取CSV文件
file_path = 'processed_inter_reevaluation.csv'
df = pd.read_csv(file_path)

# 查看数据前几行，确认书本列名
print(df.head())

# 假设书本的列名是 'book_id'，我们使用去重的方法
unique_books = df['book_id'].nunique()

# 输出去重后的书本数量
print(f"去重后的书本数量: {unique_books}")
