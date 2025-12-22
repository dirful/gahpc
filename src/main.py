# src/main.py
import numpy as np
import pandas as pd

from llm.enhanced_llm_job_gen import EnhancedLLMJobGenerator
from log.logger import get_logger
from config.config import Config
from config.feature_config import FeatureConfig
from data.db_loader import DBLoader
from data.feature_engineer import FeatureEngineer
from data.dataset_builder import DatasetBuilder
from generator.gan_trainer import GANTrainer
from llm.llm_client import LLMClient
from simulator.simulator import HPCSimulator
from agent.ppo_agent_fixed import PPOAgentFixed
from llm.llm_job_gen import LLMJobGenerator
from metrics.model_evaluator import JobQualityEvaluator

logger = get_logger(__name__)

def evaluate_synthetic_quality(real_features, synthetic_features, cfg):
    """评估合成数据质量"""

    # 确保特征对齐
    required_features = [
        'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
        'mem_mean', 'mem_std', 'mem_max',
        'machines_count', 'task_count',
        'duration_sec', 'cpu_intensity', 'mem_intensity',
        'task_density', 'cpu_cv', 'mem_cv'
    ]

    # 对齐特征列
    real_aligned = real_features[required_features].copy()
    synthetic_aligned = synthetic_features[required_features].copy()

    # 评估
    evaluator = JobQualityEvaluator(cfg)
    results = evaluator.overall_score(real_aligned, synthetic_aligned)

    # 生成可视化
    evaluator.plot_kl_bars(real_aligned, synthetic_aligned, "output/kl_bars_job.png")
    evaluator.plot_mmd_kernel(real_aligned, synthetic_aligned, "output/mmd_kernel_job.png")

    return results

def create_simulated_env_state(cfg):
    """创建模拟的环境状态"""
    return {
        'nodes': [
            {
                'node_id': i,
                'cpu_cores': 64,
                'memory_gb': 256,
                'current_cpu_util': 0.2 + 0.1 * i,
                'current_mem_util': 0.3 + 0.05 * i,
                'running_jobs': []
            }
            for i in range(min(cfg.num_nodes, 5))
        ],
        'system_load': {
            'avg_cpu_utilization': 0.25,
            'avg_memory_utilization': 0.35,
            'queue_length': 10
        },
        'job_queue': []
    }

def print_summary(cfg, gan, generated_jobs, env_sim):
    """打印运行摘要"""
    print("\n" + "="*60)
    print("HPC GenerativeAI 项目运行摘要")
    print("="*60)

    print(f"\n1. 数据配置:")
    print(f"   - 数据模式: {cfg.data_mode}")
    print(f"   - 特征数量: {len(cfg.feature_list)}")
    print(f"   - 序列长度: {cfg.seq_len}")

    print(f"\n2. GAN配置:")
    print(f"   - GAN训练轮次: {cfg.gan_epochs}")
    print(f"   - 潜在维度: {cfg.gan_latent_dim}")
    print(f"   - 输出维度: {cfg.gan_out_dim}")

    print(f"\n3. LLM配置:")
    print(f"   - 模型: {cfg.llm_model}")
    print(f"   - 生成任务数: {cfg.llm_num_generate}")

    print(f"\n4. PPO配置:")
    print(f"   - 训练轮次: {cfg.episodes}")
    print(f"   - 状态维度: {cfg.state_dim}")
    print(f"   - 动作维度: {cfg.action_dim}")

    print(f"\n5. 模拟器配置:")
    print(f"   - 节点数: {cfg.num_nodes}")
    print(f"   - 每个节点最大作业数: {cfg.max_jobs_per_node}")

    print(f"\n6. 生成结果:")
    print(f"   - 生成的作业数: {len(generated_jobs)}")

    if generated_jobs:
        print(f"   - 作业示例:")
        for i, job in enumerate(generated_jobs[:3]):
            print(f"     作业{i+1}: CPU={job.get('cpu', 0):.2f}, "
                  f"MEM={job.get('mem', 0):.2f}, "
                  f"Duration={job.get('duration', 0):.0f}s")

    if env_sim and hasattr(env_sim, 'get_metrics'):
        try:
            metrics = env_sim.get_metrics()
            print(f"\n7. 模拟器指标:")
            print(f"   - 完成作业数: {metrics.get('total_jobs_completed', 0)}")
            print(f"   - 总奖励: {metrics.get('total_reward', 0):.2f}")
            print(f"   - 队列长度: {metrics.get('queue_length', 0)}")
        except Exception as e:
            print(f"\n7. 模拟器指标: 获取失败 - {e}")

    print("="*60 + "\n")

def main():
    logger.info("=== HPC GenerativeAI Project Start ===")

    # 加载配置
    cfg = Config()

    # 根据数据模式设置特征
    feature_config = FeatureConfig.get_feature_config(cfg.data_mode)
    if feature_config['feature_list']:
        cfg.feature_list = feature_config['feature_list']
    cfg.seq_len = feature_config['seq_len']

    logger.info(f"数据模式: {cfg.data_mode}")
    logger.info(f"特征数量: {len(cfg.feature_list) if cfg.feature_list else '动态'}")
    logger.info(f"序列长度: {cfg.seq_len}")

    # 1. 加载数据
    db = DBLoader(cfg)

    try:
        if cfg.data_mode == 'time_series':
            logger.info("加载时间序列数据...")

            # 获取job统计特征
            job_features = db.extract_job_level_features_fast(
                sample_limit=cfg.sample_limit
            )

            if job_features.empty:
                logger.error("无法提取job统计特征")
                return

            # 获取时间序列数据
            job_ids = job_features.index.tolist()[:cfg.max_jobs_ts]
            window_df = db.extract_windowed_time_series_fast(
                job_ids=job_ids,
                window_seconds=cfg.window_seconds,
                max_jobs=len(job_ids)
            )

            if window_df.empty:
                logger.error("无法提取时间序列数据")
                return

            # 构建序列
            sequences, meta_df = db.build_sequences(
                window_df=window_df,
                job_df=job_features,
                seq_len=cfg.seq_len,
                min_windows=cfg.min_windows
            )

            logger.info(f"时间序列数据形状: {sequences.shape}")

            # 为特征工程准备数据
            if len(sequences.shape) == 3:
                N, T, F = sequences.shape
                # 动态设置特征列表
                if cfg.feature_list is None:
                    cfg.feature_list = [f'feature_{i}' for i in range(F)]
                    logger.info(f"动态设置特征列表: {F}个特征")

                # 创建DataFrame用于统计
                flat_data = sequences.reshape(N * T, F)
                df_for_stats = pd.DataFrame(flat_data, columns=cfg.feature_list[:F])
            else:
                df_for_stats = pd.DataFrame()

        else:  # job_stats 模式
            logger.info("加载job统计特征数据...")
            df = db.extract_job_level_features_fast(
                sample_limit=cfg.sample_limit
            )

            if df.empty:
                logger.error("无法加载数据")
                return

            df_for_stats = df

    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return

    # 2. 特征工程
    try:
        fe = FeatureEngineer(cfg)
        features = fe.transform(df_for_stats, data_mode=cfg.data_mode)

        # 如果特征列表为空，使用DataFrame的列
        if not cfg.feature_list or cfg.feature_list is None:
            cfg.feature_list = features.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"从数据中提取特征列表: {len(cfg.feature_list)}个特征")

        logger.info(f"特征工程完成，数据形状: {features.shape}")

    except Exception as e:
        logger.error(f"特征工程失败: {e}", exc_info=True)
        return

    # 3. 构建训练集
    try:
        builder = DatasetBuilder(cfg)
        seq_data = builder.build_sequences(features)

        logger.info(f"训练数据形状: {seq_data.shape}")

        # 设置PPO状态维度
        if seq_data.shape[-1] > 0:
            cfg.state_dim = seq_data.shape[-1]
            logger.info(f"设置PPO状态维度: {cfg.state_dim}")

    except Exception as e:
        logger.error(f"构建训练集失败: {e}", exc_info=True)
        return

    # 4. GAN 训练
    gan = None
    if cfg.enable_gan:
        try:
            logger.info("训练GAN...")

            # 准备GAN训练数据
            if len(seq_data.shape) == 3:
                gan_train_data = builder.build_numpy_from_sequences(seq_data)
            else:
                gan_train_data = seq_data

            # 确定特征维度
            if len(gan_train_data.shape) == 2:
                feature_dim = gan_train_data.shape[1]
            elif len(gan_train_data.shape) == 3:
                feature_dim = gan_train_data.shape[2]
            else:
                feature_dim = len(cfg.feature_list) if cfg.feature_list else 15

            # 确保配置正确
            cfg.gan_out_dim = feature_dim
            logger.info(f"设置GAN输出维度为: {cfg.gan_out_dim}")

            # 创建GAN训练器
            gan = GANTrainer(cfg)

            if gan_train_data.size > 0:
                gan.train(gan_train_data, epochs=cfg.gan_epochs)
                logger.info("GAN训练完成")

                # 生成样本
                gan_samples = gan.sample(n=cfg.gan_sample_n)
                logger.info(f"GAN生成 {len(gan_samples)} 个样本")
            else:
                logger.warning("GAN训练数据为空，跳过训练")
                gan_samples = None

        except Exception as e:
            logger.error(f"GAN训练失败: {e}", exc_info=True)
            gan_samples = None

    # 5. LLM 生成任务
    generated_jobs = []

    if cfg.enable_llm:
        try:
            logger.info("使用LLM生成任务...")

            # 获取统计信息
            stats = fe.get_stats(features)

            # 获取GAN样本
            gan_samples = None
            if gan:
                gan_samples = gan.sample(n=cfg.gan_sample_n)
                logger.info(f"GAN生成 {len(gan_samples)} 个样本")

            # 创建模拟器获取环境状态
            env_state = {}
            try:
                # 创建一个模拟器来获取环境状态
                env_sim_state = HPCSimulator(cfg)

                # 如果有历史数据，可以模拟一些作业
                if gan_samples:
                    # 添加一些GAN样本到模拟器，让环境状态更真实
                    env_sim_state.add_jobs(gan_samples[:3])

                # 检查是否有export_state方法
                if hasattr(env_sim_state, 'export_state'):
                    env_state = env_sim_state.export_state()
                else:
                    # 手动创建状态
                    env_state = {
                        'nodes': [
                            {
                                'node_id': i,
                                'cpu_cores': 64,
                                'memory_gb': 256,
                                'cpu_utilization': 0.2,
                                'mem_utilization': 0.3
                            }
                            for i in range(cfg.num_nodes)
                        ],
                        'system_load': {
                            'avg_cpu_utilization': 0.25,
                            'avg_memory_utilization': 0.35,
                            'queue_length': 5
                        }
                    }

                logger.info(f"环境状态: {len(env_state.get('nodes', []))}个节点, "
                            f"平均CPU利用率: {env_state.get('system_load', {}).get('avg_cpu_utilization', 0):.1%}")

            except Exception as e:
                logger.warning(f"获取环境状态失败: {e}")
                # 使用函数创建模拟的环境状态
                env_state = create_simulated_env_state(cfg)
                logger.info("使用模拟环境状态")

            # 初始化LLM
            llm_client = LLMClient(
                model=cfg.llm_model,
                host=cfg.ollama_host
            )

            # llm_gen = LLMJobGenerator(cfg, llm_client, gan_trainer=gan)
            #
            # # 生成任务
            # generated_jobs = llm_gen.generate_job(
            #     stats=stats,
            #     gan_samples=gan_samples,
            #     env_state=env_state,
            #     num=cfg.llm_num_generate
            # )

            llm_gen = EnhancedLLMJobGenerator(cfg, llm_client, gan_trainer=gan)
            synthetic_features = llm_gen.generate_job_level(stats, num=100)

            # 检查生成的质量
            if generated_jobs:
                # 检查多样性
                # unique_count = len(set(
                #     f"{j.get('cpu', 0):.2f},{j.get('mem', 0):.2f},{j.get('duration', 0):.0f}"
                #     for j in generated_jobs
                # ))
                #
                # logger.info(f"LLM生成 {len(generated_jobs)} 个作业，其中 {unique_count} 个是独特的")
                #
                # if unique_count < len(generated_jobs) / 2:
                #     logger.warning("生成的作业多样性不足，将进行增强")
                #     generated_jobs = llm_gen._ensure_diversity(generated_jobs, len(generated_jobs))
                print("=== 生成任务质量评估 ===")
                required_features = [
                    'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
                    'mem_mean', 'mem_std', 'mem_max',
                    'machines_count', 'task_count',
                    'duration_sec', 'cpu_intensity', 'mem_intensity',
                    'task_density', 'cpu_cv', 'mem_cv'
                ]

                real_df = features[required_features].copy()
                fake_df = synthetic_features[required_features].copy()
                eval_result = evaluate_synthetic_quality(real_df, fake_df, cfg)
                # evaluator = JobQualityEvaluator(cfg)
                #
                # eval_result = evaluator.overall_score(
                #     real_df,                  # 真实 HPC job 分布
                #     fake_df
                # )

                logger.info(eval_result["markdown"])  # 再打印一次，也可写文件

                # 保存可视化
                # evaluator.plot_kl_bars(real_df, fake_df, "output/kl_bars.png")
                # evaluator.plot_mmd_kernel(real_df, fake_df, "output/mmd_kernel.png")
                # evaluator.plot_roc(real_df, fake_df, model="rf", save_path="output/roc_rf.png")
                print(eval_result)

        except Exception as e:
            logger.error(f"LLM生成失败: {e}", exc_info=True)
            # 使用后备方案
            # generated_jobs = llm_gen._create_diverse_default_jobs(cfg.llm_num_generate)
            logger.info(f"使用多样化默认作业: {len(generated_jobs)} 个")

    # 6. 强化学习训练
    env_sim = None
    if cfg.enable_ppo and generated_jobs:
        try:
            logger.info("开始强化学习训练...")

            # 创建模拟器
            try:
                env_sim = HPCSimulator(cfg)
            except Exception as e:
                logger.error(f"创建模拟器失败: {e}")
                # 创建简单的模拟器作为后备
                env_sim = HPCSimulator(cfg)
                logger.warning("使用简单模拟器")

            # 添加生成的任务到环境
            if hasattr(env_sim, 'add_jobs') and generated_jobs:
                env_sim.add_jobs(generated_jobs)
                logger.info(f"添加 {len(generated_jobs)} 个任务到环境")

            # 保存初始状态（包含作业）
            initial_state = None
            try:
                # 尝试获取初始状态，但处理可能的失败
                if hasattr(env_sim, '_get_state'):
                    initial_state = env_sim._get_state()
                elif hasattr(env_sim, 'get_state'):
                    initial_state = env_sim.get_state()
            except Exception as e:
                logger.warning(f"获取初始状态失败: {e}")
                initial_state = None

            agent = PPOAgentFixed(cfg)

            # 训练循环
            for episode in range(cfg.episodes):
                logger.info(f"Episode {episode + 1}/{cfg.episodes}")

                # 重置环境但不重置作业队列
                if episode == 0 and initial_state is not None:
                    # 第一个episode使用已添加作业的环境
                    # 使用agent的_extract_state_features方法提取状态
                    if hasattr(agent, '_extract_state_features'):
                        state = agent._extract_state_features(initial_state)
                    else:
                        # 如果agent没有该方法，尝试简单转换
                        try:
                            state = np.array(list(initial_state.values())).flatten()[:30]
                        except:
                            state = np.zeros(30)

                    if hasattr(env_sim, 'current_step'):
                        env_sim.current_step = 0
                    if hasattr(env_sim, 'total_reward'):
                        env_sim.total_reward = 0
                    logger.info("使用已添加作业的初始状态")
                else:
                    # 后续episode：重置环境并重新添加作业
                    raw_state = env_sim.reset()
                    # 提取状态特征
                    if hasattr(agent, '_extract_state_features'):
                        state = agent._extract_state_features(raw_state)
                    else:
                        state = raw_state

                    if hasattr(env_sim, 'add_jobs') and generated_jobs:
                        env_sim.add_jobs(generated_jobs)
                    logger.info("重置环境并重新添加作业")

                episode_reward = 0
                done = False
                step_count = 0

                # 在PPO训练循环中
                while not done and step_count < 100:
                    # 选择动作
                    action, log_prob, value = agent.act(state)

                    # 简化：直接传递状态给模拟器
                    next_state_raw, reward, done, info = env_sim.step(action)

                    # 直接使用原始状态，让agent自己提取特征
                    agent.remember(state, action, reward, next_state_raw, done, log_prob, value)

                    # 更新状态（使用agent提取的特征）
                    state = agent._extract_state_features(next_state_raw)
                    episode_reward += reward
                    step_count += 1

                    # 定期学习
                    if step_count % 10 == 0:
                        agent.learn()

                # 最终学习
                agent.learn()

                logger.info(f"Episode {episode + 1} 完成，奖励: {episode_reward:.2f}, 步数: {step_count}")

                # 显示环境状态
                if hasattr(env_sim, 'render'):
                    env_sim.render()

            logger.info("强化学习训练完成")

            # 显示最终性能指标
            if hasattr(env_sim, 'get_metrics'):
                try:
                    metrics = env_sim.get_metrics()
                    logger.info(f"最终性能指标: {metrics}")
                except Exception as e:
                    logger.error(f"获取指标失败: {e}")

        except Exception as e:
            logger.error(f"强化学习训练失败: {e}", exc_info=True)

        # 打印摘要
        try:
            print_summary(cfg, gan, generated_jobs, env_sim)
        except Exception as e:
            logger.error(f"打印摘要失败: {e}")
            print("\n简单摘要:")
            print(f"GAN训练: {'完成' if gan else '未完成'}")
            print(f"生成任务数: {len(generated_jobs) if generated_jobs else 0}")
            print(f"模拟器: {'可用' if env_sim else '不可用'}")

    logger.info("=== 项目运行完成 ===")

if __name__ == "__main__":
    main()