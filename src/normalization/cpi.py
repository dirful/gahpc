import numpy as np

def process_cpi_special(cpi_data, target_mean=0.0, target_std=0.05):
    """
    特殊处理CPI特征，将其幅度调整到与其他特征匹配
    修改：针对你的数据特征调整参数
    """
    print("\n=== CPI特征特殊处理 ===")

    # 1. 原始统计
    print(f"原始CPI: mean={cpi_data.mean():.4f}, std={cpi_data.std():.4f}")
    print(f"原始范围: [{cpi_data.min():.2f}, {cpi_data.max():.2f}]")
    print(f"原始分位数: 1%={np.percentile(cpi_data, 1):.2f}, "
          f"50%={np.percentile(cpi_data, 50):.2f}, "
          f"99%={np.percentile(cpi_data, 99):.2f}")

    # 2. 对数变换（处理右偏）
    cpi_log = np.log1p(cpi_data)  # log(1+x)
    print(f"log1p后: mean={cpi_log.mean():.4f}, std={cpi_log.std():.4f}")

    # 3. 稳健标准化（使用中位数和IQR，避免异常值影响）
    median = np.median(cpi_log)
    q75, q25 = np.percentile(cpi_log, [75, 25])
    iqr = q75 - q25

    print(f"中位数: {median:.4f}, IQR: {iqr:.4f}")

    # 如果有足够的分布，使用IQR
    if iqr > 0:
        cpi_normalized = (cpi_log - median) / iqr
        print(f"IQR标准化后: mean={cpi_normalized.mean():.4f}, std={cpi_normalized.std():.4f}")
    else:
        # 使用标准差
        cpi_normalized = (cpi_log - cpi_log.mean()) / (cpi_log.std() + 1e-8)
        print(f"标准差标准化后: mean={cpi_normalized.mean():.4f}, std={cpi_normalized.std():.4f}")

    # 4. 关键修改：调整到与其他特征匹配的幅度
    # 你的其他特征std≈0.01-0.1，所以target_std设为0.05
    current_mean = cpi_normalized.mean()
    current_std = cpi_normalized.std()

    # 如果std太小或太大，进行缩放
    if current_std > 0:
        # 计算缩放因子
        scale_factor = target_std / current_std
        print(f"缩放因子: {scale_factor:.4f} (target_std={target_std}, current_std={current_std:.4f})")

        # 调整到目标幅度
        cpi_adjusted = (cpi_normalized - current_mean) * scale_factor + target_mean
    else:
        cpi_adjusted = cpi_normalized

    print(f"幅度调整后: mean={cpi_adjusted.mean():.6f}, std={cpi_adjusted.std():.6f}")

    # 5. 更温和的裁剪（保留分布信息）
    # 检查当前分布范围
    p001 = np.percentile(cpi_adjusted, 0.1)
    p999 = np.percentile(cpi_adjusted, 99.9)

    # 设置更宽松的边界
    # 你的其他特征范围大约[-0.1, 0.5]，所以CPI也可以在这个范围
    clip_min = max(p001, -0.5)  # 不要剪掉太多负值
    clip_max = min(p999, 1.0)   # 稍微宽松一点

    print(f"建议clip范围: [{clip_min:.4f}, {clip_max:.4f}]")

    cpi_final = np.clip(cpi_adjusted, clip_min, clip_max)

    # 6. 最终统计
    print(f"最终CPI: mean={cpi_final.mean():.6f}, std={cpi_final.std():.6f}")
    print(f"最终范围: [{cpi_final.min():.6f}, {cpi_final.max():.6f}]")
    print(f"最终分位数: 1%={np.percentile(cpi_final, 1):.6f}, "
          f"50%={np.percentile(cpi_final, 50):.6f}, "
          f"99%={np.percentile(cpi_final, 99):.6f}")

    # 计算裁剪比例
    n_clipped_low = (cpi_adjusted < clip_min).sum()
    n_clipped_high = (cpi_adjusted > clip_max).sum()
    total = cpi_adjusted.size

    print(f"裁剪统计: {n_clipped_low/total*100:.2f}%裁剪到下限, "
          f"{n_clipped_high/total*100:.2f}%裁剪到上限")

    return cpi_final