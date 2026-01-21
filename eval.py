import torch
import os
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import List

# 必须在导入 mujoco 相关的库之前设置，确保远程服务器离屏渲染
os.environ['MUJOCO_GL'] = 'egl'

# 导入你训练脚本中的配置类和工具函数
# 假设你的训练脚本名为 demo.py
from demo import (
    GymnasiumVecEnv, 
    TrainConfig, 
    CustomActorCritic, 
    _resolve_device, 
    _build_train_cfg
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

def evaluate(model_path: str, num_episodes: int = 5):
    # 1. 基础配置初始化
    cfg = TrainConfig()
    device = _resolve_device("auto")
    # 强制将环境 ID 设为 Hopper (或你训练时用的环境)
    env_id = cfg.env.env_id 
    train_cfg = _build_train_cfg(cfg)
    
    # 2. 实例化带录制功能的测试环境
    # 我们直接修改 GymnasiumVecEnv 的创建逻辑，使其支持渲染和录制
    print(f"正在创建测试环境: {env_id}...")
    
    # 创建一个单环境的 VecEnv 适配器
    # 注意：我们手动在内部添加 RecordVideo 包装器
    class EvalVecEnv(GymnasiumVecEnv):
        def __init__(self, env_id, num_envs, seed, device):
            super().__init__(env_id, num_envs, seed, device)
            # 替换掉内部的单环境，增加渲染和录制功能
            self.env.envs[0] = gym.make(env_id, render_mode="rgb_array")
            self.env.envs[0] = gym.wrappers.RecordVideo(
                self.env.envs[0],
                video_folder="./eval_videos",
                # step_trigger 设置为 0 表示每个 episode 开始时都触发录制
                episode_trigger=lambda x: True, 
                name_prefix=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    env = EvalVecEnv(env_id, num_envs=1, seed=cfg.env.seed, device=device)

    # 3. 动态注入自定义模块并初始化 Runner
    import rsl_rl.modules as rsl_modules
    import rsl_rl.runners.on_policy_runner as rsl_runner
    rsl_modules.CustomActorCritic = CustomActorCritic
    rsl_runner.CustomActorCritic = CustomActorCritic
    
    # 初始化 Runner（不记录日志）
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    
    # 加载指定的权重文件
    if os.path.exists(model_path):
        print(f"成功加载权重: {model_path}")
        runner.load(model_path)
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return

    # 获取策略并切换到评价模式（关闭随机噪声探索）
    policy = runner.alg.policy
    policy.eval()

    # 4. 执行测试循环
    print(f"开始推理测试，共执行 {num_episodes} 个回合...")
    
    for ep in range(num_episodes):
        obs_td = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        
        while not done:
            with torch.no_grad():
                # 使用 act_inference 获取确定性动作
                actions = policy.act_inference(obs_td)
            
            # 环境步进
            obs_td, reward, done_tensor, infos = env.step(actions)
            
            ep_reward += reward.item()
            step_count += 1
            # dones_t 在 GymnasiumVecEnv 中是 bool tensor
            done = done_tensor.any().item()
            
        print(f"回合 {ep+1}: 步数 = {step_count}, 总奖励 = {ep_reward:.2f}")

    # --- 关键步骤：关闭环境以完成视频文件写入 ---
    env.env.close()
    print(f"\n测试完成！视频已保存至目录: {os.path.abspath('./eval_videos')}")

if __name__ == "__main__":
    # 请确保此路径指向你实际存在的模型文件
    MODEL_PATH = "/data/hopper/runs/rsl_rl_hopper/20260120_142619/model_999.pt"
    
    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"警告: 默认路径 {MODEL_PATH} 不存在，请修改脚本中的 MODEL_PATH。")
    else:
        evaluate(MODEL_PATH, num_episodes=1)