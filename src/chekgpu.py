import torch

# 1. 检查 PyTorch 能否识别到 NVIDIA 显卡
gpu_available = torch.cuda.is_available()

print(f"--- 诊断报告 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 是否可用: {gpu_available}")

if gpu_available:
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("结论: 你目前正在使用 CPU 版本，或者驱动/CUDA 没配置好。")