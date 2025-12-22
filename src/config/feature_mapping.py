# config/feature_mapping.py

FEATURE_MAPPING = {
    # LLM基础特征 → job-level特征 的映射关系
    'llm_to_job': {
        'cpu': ['cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th'],
        'mem': ['mem_mean', 'mem_std', 'mem_max'],
        'duration': ['duration_sec', 'cpu_intensity', 'mem_intensity'],
        'submission_time': ['start_time'],
        'disk_io': [],  # 暂不映射到job-level
    },

    # 必需的job-level特征（来自DBLoader）
    'required_job_features': [
        'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
        'mem_mean', 'mem_std', 'mem_max',
        'machines_count', 'task_count',
        'duration_sec', 'cpu_intensity', 'mem_intensity',
        'task_density', 'cpu_cv', 'mem_cv'
    ]
}