# src/log/logger.py
import logging
import os
from datetime import datetime

# 全局缓存：避免重复创建 logger
_LOGGER_POOL = {}


def get_logger(name="default", level=logging.INFO, log_to_file=True):
    """
    获取项目通用 logger。
    - 支持 console + file 输出
    - 支持子模块命名，如 "ppo.policy"
    - 防止重复添加 handler
    """

    # 若 logger 已存在，直接返回
    if name in _LOGGER_POOL:
        return _LOGGER_POOL[name]

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 避免重复打印

    # -------------------------
    # Console 输出
    # -------------------------
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(ch)

    # -------------------------
    # File 输出
    # -------------------------
    if log_to_file:
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 按名称分开日志，例如 gan.log、ppo.log、sim.log
        log_file = os.path.join(log_dir, f"{name}.log")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        ))
        logger.addHandler(fh)

    # 缓存
    _LOGGER_POOL[name] = logger
    return logger


# 用于快速获取预配置 logger
def get_debug_logger(name="debug"):
    return get_logger(name, level=logging.DEBUG)


def get_info_logger(name="info"):
    return get_logger(name, level=logging.INFO)
