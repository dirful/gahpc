# 使用默认参数运行
python main.py

# 指定参数运行
python main.py --sample_size 100000 --n_clusters 6 --optimize

# 快速运行（小样本，不优化）
python main.py --sample_size 20000 --n_clusters 4 --no_optimize

# 查看所有选项
python main.py --help