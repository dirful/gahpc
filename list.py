import os

# 定义src目录路径（可根据实际路径调整，这里是当前工作目录下的src）
src_dir = os.path.join(os.getcwd(), "src")

# 存储src下的目录列表
src_dirs = []

# 遍历src目录
if os.path.exists(src_dir) and os.path.isdir(src_dir):
    # 遍历目录下所有内容
    for item in os.listdir(src_dir):
        item_path = os.path.join(src_dir, item)
        # 判断是否为目录
        if os.path.isdir(item_path):
            src_dirs.append(item_path)  # 存储完整路径，也可只存目录名item
else:
    print(f"目录 {src_dir} 不存在或不是有效目录")

# 打印结果
print("src下的目录列表：")
for dir_path in src_dirs:
    print(dir_path)
