---
tags:
  - note
  - rl
  - camera_inteligence
  - with_wfs
---

# W1 2026.1.14 ~ 1.21

---
What is the problem?
- RL-based Next-best-view prediction for indoor 3d scene reconstruction in simulation environment

## 1.0 TODO
- RL: PPO (rsl_rl, https://github.com/leggedrobotics/rsl_rl)【了解流程
- Next-best-view related work / baseline: GLEAM (https://xiao-chen.tech/gleam/)【需要复现结果】
	- pipeline
	- related-work (active reconstruction )
- simulation environment: ManiSkill3 (https://github.com/haosulab/ManiSkill)【了解】
- Feed-forward 3d reconstruction: VGGT (https://vgg-t.github.io/) 【bilibili talk (https://www.bilibili.com/video/BV1UyunzLE4J)】
- Indoor 3d scene: ReplicaCAD, and AI2THOR , HSSD (https://3dlg-hcvc.github.io/hssd/) 【了解，scale up训练集】

> 1.15 讨论更新
- VGGT 
	- 下游任务相关工作 LVSM（新视角合成）
	- limitation and improvements
		- py3: 不同于vggt首图世界坐标
		- mapanything：任意模态输入
- GLEAM
	- related works！
		- 更新：也许可以往后放放

> 1.16 讨论更新
- rsl_rl ppo 训练一个标准强化学习任务 dm-control walker-walk (reward curve 上涨到快1000)
- rsl_rl ppo on gleam-like task
	- input: rgb / depth image
	- output: camera view
	- policy: cnn （重点：处理不定长序列）（ possible solution：max pooling）(vggt)
	- reward: coverage (depth -> point cloud -> voxelize)
	- 理解：cnn 作为 baseline 的方法

---

## 1.1 RL: TRPO & PPO

#### 策略梯度定理 --- REINFORCE with baseline --- Actor-Critic(AC) ---GAE

**策略梯度定理**
$$J(\theta) = \mathbb{E}_{\pi_\theta} [R(\tau)]$$
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi}(s, a) ]$$
**REINFORCE with baseline**
- REINFORCE 即使用 monte carlo 估计动作价值Q，增加基线值 b(s) 不改变结果（导数求和可以推出0），有效减少方差
$$\nabla J(\theta) = \sum_{s} \mu_{\pi}(s) \sum_{a} \left( q_{\pi}(s,a) - b(s) \right) \nabla_{\theta} \pi(a|s,\theta)$$
**Actor-Critic**
- 两个接受状态 s 作为参数的神经网络
	- **Actor（策略网络）**：负责输出动作概率 $\pi(a|s)$。
	- **Critic（价值网络）**：负责输出状态价值 $V(s)$。
- 用 Critic 网络估计的值作为baseline，更新Actor的参数 $\theta$
$$\theta_{t+1} = \theta_t + \alpha \gamma^t (G_{t:t+1} - \hat{v}(S_t, w)) \nabla_\theta \log \pi(A_t | S_t, \theta)$$
$$\dots = \theta_t + \alpha \gamma^t (R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)) \nabla_\theta \log \pi(A_t | S_t, \theta)$$
$$\theta_{t+1} = \theta_t + \alpha \gamma^t \delta_t \nabla_\theta \log \pi(A_t | S_t, \theta)$$
- 在策略梯度定理中，用 monte-carlo 算法估计Q值即 REINFORCE 算法，而这里用 TD(0) 单步估计回报G

**GAE**：指数加权结合1～n步TD误差
$$A^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$$$A^{(2)} = \delta_t + \gamma \delta_{t+1}$$
$$A^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$$
$$\hat{A}_t^{GAE} = (1-\lambda) \left( A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + \dots \right)$$

#### TRPO (Trust Region Policy Optimization)

>改进自传统策略梯度算法，传统算法中通过梯度上升，更新过大会导致严重训练不稳定问题
>试图通过新旧策略之间的某种 KL-散度 差异选择优化参数

参考短片： https://www.bilibili.com/video/BV1Jc8BzwEhE?spm_id_from=333.788.videopod.sections&vd_source=819c7595b7626630d7da04517f99f477

- 策略价值 (Performance Difference Lemma)：
$$\eta(\pi') = \eta(\pi) + \sum_{s \in \mathcal{S}} \rho_{\pi'}(s)\sum_{a \in \mathcal{A}} \pi'(a|s) A_{\pi}(s, a) $$
	- 优势函数 $A$: $$A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$$
	- 新策略 $\pi'$ 未知
	- 状态转移 $\eta_{\pi'}(s)$ 未知

- 如果新旧策略足够接近，新策略真实价值 $\eta(\pi')$ 有性能下界（数学严格成立）：
$$\eta(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - C \cdot D_{TV}^{max}(\pi_{old}, \pi_{new})$$
- Trust Region:  工程上解决惩罚项太大导致严格上升区域太小，以及用KL散度近似解决TV散度难以计算的问题
$$D_{KL}^{max}(\pi_{old}, \pi_{new}) \le \delta$$
- 工程实现：
$$\text{maximize}_{\theta} \quad \hat{\mathbb{E}}_t \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]$$
$$\text{subject to} \quad \hat{\mathbb{E}}_t [D_{KL}(\pi_{\theta_{old}}(\cdot|s_t) || \pi_{\theta}(\cdot|s_t))] \le \delta$$
#### PPO (Proximal Policy Optimization)

- PPO-Penalty: 没什么特别的，把TRPO中约束改成惩罚项
$$\text{maximize}_{\theta} \quad \mathbb{E}_t \left[ \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t - \beta D_{KL}(\pi_{\theta_{old}} || \pi_{\theta}) \right]$$
- PPO-Clip：既然我们希望约束新旧策略比较接近，而且目标函数（$L = \mathbb{E}[r_t(\theta) \cdot A_t]$）中仅重要性采样比（$r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\theta_{old}}}$）与约束相关，不妨直接对重要性采样比进行**截断**：  
$$J^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]$$
>为什么要再嵌套一层 $min$ ？
>如果目标策略是好策略（$A_t > 0$），clip保证策略更新的稳定
>如果$A_t < 0$，min 取前一项保证有梯度用来回退

## 1.2 RL: rsl_rl

> repo： https://github.com/leggedrobotics/rsl_rl
> arXiv： https://arxiv.org/pdf/2509.10771

![[rsl_rl.png]]

- **Features**:
	- PPO and Teacher-Student Learning
	- rare ML algorithms
	- Runners + Algorithms + Networks

> 一些代码切片 

- rsl/rsl/algorithms/ppo.py
```
def compute_returns(self, obs: TensorDict) -> None:
	...
	
	for step in reversed(range(st.num_transitions_per_env)):
		...
		
		# TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
		delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
		
		# Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
		advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
		
		# Return: R_t = A(s_t, a_t) + V(s_t)
		st.returns[step] = advantage + st.values[step]
		
	# Compute the advantages
	st.advantages = st.returns - st.values
	
	# Normalize the advantages if per minibatch normalization is not used
	if not self.normalize_advantage_per_mini_batch:
		st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
```
adaptive kl
```
if kl_mean > self.desired_kl * 2.0:
	self.learning_rate = max(1e-5, self.learning_rate / 1.5)
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
	self.learning_rate = min(1e-2, self.learning_rate * 1.5)
```
clip
```
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
	ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

clip 很直观，个人理解 adaptive kl 形式上感觉属于 penalty 的一个改进版本，可能实践上比较好用


---
## 1.3 ManiSkill3

> simulation environment
> repo: https://github.com/haosulab/ManiSkill
> arXiv: https://arxiv.org/pdf/2410.00425

> fs哥：maniskill3不需要都看，你可以关注他是怎么 load scene，然后 scene 的 data format 是什么样的

> tutorial (official):  https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb

```
This notebooks covers:

1. How to run environments/tasks, modify them, record them, and teleoperate them. (GPU parallelized rendering of RGB-D data achieves 3000+ FPS on even Google Colab's free tier!)
   
2. Some simple state/visual based RL code/workflows leveraging the GPU accelreated simulator. Solve robotics tasks in minutes instead of hours/days!
   
3. Working with demonstration/offline data, how to load and replay and convert them
   
4. Heterogeneous Parallel Simulation. Showcasing simulating tasks where the number of objects or the articulations are completely different between each scene
   
5. Playing with more robots not included out of the box.
   
6. Running motion planning solutions to solve tasks.
```

```
while not done:
	# env.action_space.sample(): 随机采样动作 
	# 注意：这里的 action 也是矢量化的，一次性包含 512 条指令 
	
	obs, rew, terminated, truncated, info = env.step(torch.from_numpy( env.action_space.sample())) 
	
	done = (terminated | truncated).any()
```

```
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

dataset = ManiSkillTrajectoryDataset(dataset_file="demos/PegInsertionSide-v1/motionplanning/trajectory.h5")

data = dataset[150]
for k, v in data.items():
print(k, v)
```

```
# Can be any env_id from the list of Rigid-Body envs: https://maniskill.readthedocs.io/en/latest/tasks/index.html

env_id = "PickCube-v1" #@param ['PickCube-v1', 'PegInsertionSide-v1', 'StackCube-v1']

  

# choose an observation type and space, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for details

obs_mode = "rgb+depth+segmentation" #@param can be one of ['pointcloud', 'rgb+depth+segmentation', 'state_dict', 'state']

  

# choose a controller type / action space, see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html for a full list

control_mode = "pd_joint_delta_pos" #@param can be one of ['pd_ee_delta_pose', 'pd_ee_delta_pos', 'pd_joint_delta_pos', 'arm_pd_joint_pos_vel']

  

reward_mode = "dense" #@param can be one of ['sparse', 'dense']

  

robot_uids = "panda" #@param can be one of ['panda', 'fetch']

  

# create an environment with our configs and then reset to a clean state

env = gym.make(env_id,

num_envs=4,

obs_mode=obs_mode,

reward_mode=reward_mode,

control_mode=control_mode,

robot_uids=robot_uids,

enable_shadow=True # this makes the default lighting cast shadows

)

obs, _ = env.reset()

print("Action Space:", env.action_space)
```

```
from mani_skill.trajectory.utils import index_dict, dict_to_list_of_dicts
index_dict(traj_h5["env_states"], 23) # select the 23rd element of all values
---
{'actors': {'table-workspace': array([-0.12     ,  0.       , -0.9196429,  0.7071068,  0.       ,
          0.       ,  0.7071068,  0.       ,  0.       ,  0.       ,
          1.       ,  0.       ,  0.       ], dtype=float32),
  'peg_0': array([-7.4812392e-04, -6.9533452e-02,  2.2151815e-02,  8.3441180e-01,
          2.7385606e-06,  1.7302322e-06,  5.5114156e-01,  0.0000000e+00,
          0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00], dtype=float32),
  'box_with_hole_0': array([0.01340787, 0.2980187 , 0.10695254, 0.6694303 , 0.        ,
         2.        , 0.7428749 , 0.        , 0.        , 0.        ,
         3.        , 0.        , 0.        ], dtype=float32)},
 'articulations': {'panda_wristcam': array([-6.1500001e-01,  7.2759576e-11, -1.4901161e-08,  1.0000001e+00,
          0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
          0.0000000e+00, -1.1801933e-01,  5.5470723e-01, -9.5967531e-02,
         -2.0037885e+00,  7.6059960e-02,  2.5535121e+00, -6.5118825e-01,
          3.9999980e-02,  3.9999846e-02,  4.6332176e-03,  2.3060279e-02,
          8.9918636e-03,  4.9652411e-03,  5.4766191e-03,  2.3511205e-02,
         -2.7752956e-03, -7.1137889e-05,  7.7287856e-05], dtype=float32)}}
```

### 搭建自己的

> source code: https://github.com/haosulab/ManiSkill/tree/main/mani_skill/utils

> 以下部分贴自 utils/readme

These are various functions/tooling that help make it easy for ManiSkill to work, as well as build your own tasks, train on them, and evaluate on them.

Description of the main modules are as follows:

`building/` - All useful utility code for building a task and/or scene. Includes functions for loading assets and articulations from various datasets, and randomization functions useful for randomizing task initialization. `scene_builder/` - Contains code relating to the `SceneBuilder` class and provides some prebuilt scene builders for a standard table top scene, [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/), as well as scenes from [AI2THOR](https://ai2thor.allenai.org/) via the [HSSD dataset](https://huggingface.co/datasets/hssd/ai2thor-hab).

`geometry/` - Various functions for working with geometry, from sampling primitive shapes to getting axis-aligned bounding boxes of articulations/actors.

`wrappers/` - Wrapper classes that provide additional functionality such as recording videos/episodes, modifying observation spaces, as well as adapting the environment API so that RL libraries such as [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) work out of the box.

`visualization/` - Visualization tools

`gym_utils.py` - Various utilities for working with the gymnasium/gym API

`common.py` - A ton of fairly common utilities, including those that are often used for reward functions, success evaluation, as well as working with nested dictionaries.


---

## 1.4 VGGT (Visual *Geometry(?)* Grounded Transformer)

>Feed-forward 3d reconstruction: VGGT 
>repo: https://vgg-t.github.io/
>arXiv: https://arxiv.org/pdf/2503.11651
>bilibili talk: https://www.bilibili.com/video/BV1UyunzLE4J 49min ✅

### Framework:

![[VGGT.png]]

- formulation：$$f(I_i)_{i=1}^N = (g_i, D_i, P_i, T_i)_{i=1}^N$$
	- input：n张图
	- output：得到**相机参数g**、**深度估计D**、**点云P**和**点匹配T**
		- $g_i = [q, t, f]$ $\in \mathbb{R}^9$， $q \in \mathbb{R}^4$ 表示旋转，$t \in \mathbb{R}^3$ 表示光心坐标，$f \in \mathbb{R}^2$表示相机横纵视场大小（焦距）
		- $D_i \in \mathbb{R}^{H \times W}$  深度图
		- $P_i \in \mathbb{R}^{H \times W \times 3}$ 点图（，理论上这里点图可以通过深度图和相机参数直接推算，但不这样做，可能是为了具有空间直觉）
		- $T_i \in \mathbb{R}^{C \times H \times W}$ 追踪特征 (3D Tracking Features)，高维特征描述符
- feature
	- 对称性：except frame 1 （提供光心位置）
	- Over-complete prediction
	- **No 3D inductive biases** (philosophy!)


#### detail - 1：用 q 表示旋转
> 刚体旋转参考资料： https://zhuanlan.zhihu.com/p/94584704 https://zhuanlan.zhihu.com/p/94853040

- 轴角法（表示不唯一，无法合成）
- 四元数法（使用模长为1的四元数表示三维旋转）
$$\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$$$$\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{i}\mathbf{j}\mathbf{k} = -1$$
$$\mathbf{q} = [w, x, y, z] \in \mathbb{R}^4$$
- 轴角到四元数： $w = \cos(\theta/2)$     $x = u_x \sin(\theta/2)$     $y = u_y \sin(\theta/2)$     $z = u_z \sin(\theta/2)$

#### detail - 2：追踪特征 T
> 模型有单独 module 提取对每一个帧 $i$，有 $C \times H \times W$ 维度参数 $T_i$
- 假设第一帧中有查询点坐标 $y_q = (u_q, v_q)$
- 从 $T_1$ 的对应位置取出特征向量 $\mathbf{f}_q = T_1(u_q, v_q) \in \mathbb{R}^C$
- not finished
#### framework - 1: Concat camera token & registers
> 对图像分 patch - DINO 提取特征之后，需要对 patch 特征进行一些如下处理
$$[\underbrace{t_g^i}_{相机}, \underbrace{t_{R,1}^i, \dots, t_{R,4}^i}_{寄存器}, \underbrace{t_{I,1}^i, \dots, t_{I,K}^i}_{图像Patch}]$$
- Camera tokens：随机初始化的 $\mathbb{R}^{1 \times C'}$ 相机位姿参数
- Register tokens：DINOv2 提出的寄存器 $\mathbb{R}^{4 \times C'}$，用于吸收一些无关特征、噪声等
	- 理解：attention 机制中存在 softmax，这导致无论如何行和都需要是1，但有些点（比如纯黑背景）没有什么意义，所以强行归一化可能导致 attention 产生离群值，这里放几个没用参数用于吸收。
- 第一帧独立初始化这两种 tokens，后续帧共享一样的初始化
	- 第一帧作为 coordinate frame，相机参数 $q_1 = [0, 0, 0, 1]$ $t_1 = [0, 0, 0]$

#### framework - 2: Alternating-Attention (AA Transformer)
> 传统可能使用 cross-attn，这里则是交替使用全局注意力和纯粹的帧内注意力

- frame-wise attn insights（from author‘s talk）
	- 我们需要有 *patch 来自于哪张图* 的信息，否则可能由于不同照片中的 patch 语义相似覆盖视角等差异
		- 一种可能的想法是 embedding，这会：（1）破坏除了第一张图以外的对称性；（2）折损一些泛化性（当前VGGT只用小于24帧数据训练，但可以泛化到128以上）
		- 相信模型可以通过 frame-wise 结构自发聚类一些特征信息

not finished


### Downstream Tasks & Transform Learning

#### a. Feed-forward Novel View Synthesis (NVS)
> 传统 NVS 一般需要输入图像精确的相机位姿
> VGGT 方案不利用3D表示，不需要输入图像相机位姿表示，直接预测新视角图像

- Input: 4 views, Plucker rays () to represent target viewpoints
- 用 Conv Layer 来 encode 目标视角普朗克射线图 得到所谓的，不提供其他图像的普朗克射线图

- 用 DPT 提取新视角图像

## 1.4-addition-1 LVSM

not finished




## 1.4-addition-2 $\pi^3$: Permutation-Equivariant Visual Geometry Learning
> repo: https://github.com/yyfz/Pi3
> arXiv: https://arxiv.org/pdf/2507.13347
> 参考资料: https://zhuanlan.zhihu.com/p/1930321230923306628

- 基于 VGGT 去除“归纳偏置”的思想，去除首帧的特殊地位
	- 排列等变性（Permutation Equivariance）
		- (1) 去掉所有与排列顺序有关的设计, 例如VGGT中的position encoding 
		- (2) 位姿采用相对位姿进行编码 
		- (3) 每帧预测自己单独的 pointmap (但是各帧之间尺度一致)

		- input: $S = (I_1, \dots, I_N)$, $I_i \in \mathbb{R}^{H \times W \times 3}$$$\phi(S) = ((T_1, \dots, T_N), (X_1, \dots, X_N), (C_1, \dots, C_N))$$此处，$T_i \in SE(3) \subset \mathbb{R}^{4 \times 4}$ 是相机位姿，$X_i \in \mathbb{R}^{H \times W \times 3}$ 是在其自身相机坐标系下表示的关联像素对齐 3D 点图，$C_i \in \mathbb{R}^{H \times W}$ 是 $X_i$ 的置信度图，每一个输出都与输入图像 $I_i$ 一一对应。对于任何置换 $\pi$，设 $P_\pi$ 为对序列进行排序置换的操作符。网络 $\phi$ 满足置换等变性质：
	$$\phi(P_\pi(S)) = P_\pi(\phi(S))$$$$P_\pi(\phi(S)) = (T_{\pi(1)}, \dots, T_{\pi(N)}), (X_{\pi(1)}, \dots, X_{\pi(N)}), (C_{\pi(1)}, \dots, C_{\pi(N)})$$
	- 尺度一致性：个人理解就是直接引入一个参数，先最优化这个参数再求点云的 loss
$$s^* = \arg \min_{s} \sum_{i=1}^{N} \sum_{j=1}^{H \times W} \frac{1}{z_{i,j}} \|s \hat{x}_{i,j} - x_{i,j}\|_1 \quad (4)$$$$L_{points} = \frac{1}{3NHW} \sum_{i=1}^{N} \sum_{j=1}^{H \times W} \frac{1}{z_{i,j}} \|s^* \hat{x}_{i,j} - x_{i,j}\|_1 $$
	- 相机位姿二义性：
		- 相对位姿：$\hat{T}_{i \leftarrow j} = \hat{T}_i^{-1} \hat{T}_j$
		- 旋转 $\hat{R}_{i \leftarrow j} \in SO(3)$ 和平移 $\hat{t}_{i \leftarrow j} \in \mathbb{R}^3$
		- 对于相机位姿 loss 的最终表达形式如下$$L_{cam} = \frac{1}{N(N-1)} \sum_{i \neq j} (L_{rot}(i,j) + \lambda L_{trans}(i,j)) \quad (8)$$
具体的实现细节没有详细看，基本上理解架构和这些 tricks 了


## 1.4-addition-3 MapAnything: Universal Feed-Forward Metric 3D Reconstruction
> repo: https://github.com/facebookresearch/map-anything
> arXiv:  https://arxiv.org/pdf/2509.13414

> 主要创新是一个支持多种模态输入的 3D reconstruction

![[VGGT-addition3-MapAnything.png]]








## 1.5 GLEAM (Learning Generalizable Exploration Policy for Active Mapping in Complex 3D Indoor Scenes)
#arXiv2025 #iccv #iccv2025
> sort of related work. 需要跑通看看结果
> repo: https://xiao-chen.tech/gleam/
> author's talk: https://www.bilibili.com/video/BV1oPaDzeEfo/?spm_id_from=333.337.search-card.all.click&vd_source=03a8e4a181629720c5623ae40bda6ea9
> arXiv: https://arxiv.org/pdf/2505.20294

### Paper Reading
> 主要创新：GLEAM Bench，Egocentric Mapping，Long-term Planning

#### GLEAM Bench
- 概况
	- 1152 室内 scene
	- high diversity
	- ～10 rooms per scene
	- synthesis + real scanned data
- 采集细节
	- 这里作者提到用了两个虚拟数据集和两个真实数据集，其中很多现有数据集是 “assets of ProcTHOR through the Habitat platform and Unity Engine”，这里作者写了自动化脚本把数据转化成了 mesh 形式
	- 为了解决原来长程规划的问题——表面不连续等问题导致仿真穿模，数据集中有400+复杂数据集，添加高密度杂物和几何结构复杂表面

#### GLEAM 方法
Framework: 
![[gleam1.png]]

##### task formulation
- aim:  机器人的运动策略 $\pi$ 
- observation: $o_t$ 包含相机位姿 $pose$ 和深度图 $depth$ 
- 维护的变量：
	- $M_t$ egocentric map
	- $G_t$ 世界坐标下 probabilistic global map，用于结合所有历史观测
	- $s_t$ current state

##### detail - 1: Semantic Map Representations
![[gleam2.png]]
- $M_t$ egocentric map
	- 把场景分成了若干个小块（cell），一共归为四类（传统方法只有前两类）
		- occupied
		- free
		- unknown
		- frontier

- $G_t$ 世界坐标下 probabilistic global map，用于结合所有历史观测：**点云表示!**
	- initialization
		- 注意这里其实我们有相机的内参外参 + 深度图，是可以直接反投影重建三维的
$$P_{world} = R \cdot (K^{-1} \cdot d \cdot [u, v, 1]^T) + T$$
		 - 高度过滤：只保留预设高度的点（也就是不考虑地板和天花板）
		 - 自顶向下投影，压缩回 2D Binary Occupancy Map，编码是否被占据
			 - 注意：设置阈值是为了，比如吊灯不会影响地面机器人，而椅子不会影响无人机

	 - update：用每次采集到的 $M_t$ 来更新 $G_t$
		 - 实际上会把 binary map 扩展为概率图，作为真正的 $G_t$
		 - 参考 SLAM 中的经典**占据栅格地图**算法
$$\log \text{Odd}(m_i | z_j) = \log \text{Odd}(m_i) + C$$
$$C = \log \frac{p(z_j | m_i=1)}{p(z_j | m_i=0)}$$
		- 可以将概率离散化为 occupied(1)，free(0)，unknown(0.5)
	- frontier
		- 为了提高探索倾向
		- 用一个 conv 来识别概率图中 free 和 unknown 的边界，把临街的块标记为 frontier

	- 每一轮把计算出的 $G_t$ 和 ground truth 比较计算 coverage ratio，具体细节在后面

##### detail - 2: Long-term Action Space
> 动机：1. 主要计算开销在 simulation，短程的更新仿真频率太高；2. 短程动作空间可能会导致策略 short-sighted，任务本身希望可以有更长期的环境结构理解

- 原子动作包含一定距离内可达的目标，在 agent 坐标系下参数化为向量  $(\Delta x, \Delta y, \Delta \theta)$
	- A* 搜索规划局部轨迹
	- 只在长程目标采集数据，进行仿真（关键帧）


##### detail - 3: Training
感觉没什么特别的，主要就是批次训练、随机初始化和终止条件
- 这里提到一个奇怪的事：原来的规划方法都不管碰撞？
	- 经典种植包括：最大回合长度 & 成功的阈值
	- 这里引入：碰撞、过去10步探索停滞、或者长程目标不可导航or超过预定义长度，这三个作为补充准则

##### detail - 4： Reward Function and Optimization
- 覆盖率：即时间 t 步探索出的占据和 ground truth 占据体素比例
$$CR_t = \frac{\tilde{N}_t}{N^*} \cdot 100\%$$
- **主奖励函数**直接定义成连续步覆盖率差值：
$$r^{t+1}_{CR} = CR_{t+1} - CR_t$$
- 危险探索负奖励（惩罚）：如果预测的目标点在观测区域内，且会发生碰撞，直接执行 Re-planning，但即使如此也要吃一个惩罚: $r_t^{Collision} = -1$ if collision, else 0
- 终止探索奖励：覆盖率达到 75% （90%？这块在说啥没看懂）$r_t^{Termination} = +1$ if terminate, else 0


#### GLEAM Related Works
> 主要是以coverage为目标的重建

- 论文 related works：
	- Active Mapping
		- Frontier-based exploration (FBE) policies
		- Information gain-based policies
	- Existing benchmarks
	- Generalizability
- 论文 baseline：ANM, OccAnt

> 由于这部分感觉还挺多有意思的文章，遂决定单独开一个笔记来记录一下


### 复现
- Device Requirements
	- NVIDIA RTX 3090/4090 (24GB VRAM)
	- NVIDIA Driver: 545.29.02
	- Ubuntu 20.04
	- CUDA 11.8
	- Python 3.8.12
	- PyTorch 2.0.0+cu118

#### 路径 - 1（**deprecated**） autodl - 北京B区 / 279机 fdf9468415-52741416
- 遇到的问题与解决
	- isaac gym 需要浏览器下载到本地再上传
	- `aiohttp==3.13.3` `urllib3==2.6.3` 要求太高了似乎，去掉版本要求可以正常安装

	- （未解决）sim = gym.create_sim(sim_device, graphics_device, gymapi.SIM_PHYSX, sim_params) 这一步graphicdevice必须是-1，如果是0会导致没有图形化界面报错；但如果强行设置-1则会在后续无法读入（？）图像数据，仍然报错
		- autodl 本身是无图形化界面的 docker 容器！
		- 转发远程桌面：cv.show 测试通过，然而 gym.create_sim 依旧core dumped
		- 经过 1.16 线下讨论学习，isaac gym 本身 demo 样例也无法在 autodl 容器中正确运行（core dumped）所以怀疑是库本身对图形界面比较苛刻
		- 

> 图形化界面 note: https://marmalade-iguana-367.notion.site/fc43020c3fef4c97a792d243248e590b?pvs=25 from fs哥

#### 路径 - 2 fs-desktop not finished
- 遇到的问题与解决
	- 与fs哥共用用户，但通过传入 junyi 用户参数伪装，直接在 .bashrc 跳转进文件夹 /home/fs/junyi/
	> **主机配置**相关：
	- conda： https://github.com/conda-forge/miniforge
	- ToDesk: device ID 616 697 275 password: Ul4>?458Z@.m
	- （未解决）当前卡在主机网络出了一些问题（1.16 最后情况是：链接有时候会卡，ping可以连网页，但是如果conda create环境会超时，然后多试了几次就会直接与主机断联）
		- 师兄在排查

#### 路径 - 3 闪电算力 成功！✅
> www.9gpu.com

- 其实又便宜又好，据说（确实）还有图形化界面，但就是小杂牌
- 不能翻墙，下载到本地再上传
```
Host ry3.9gpu.com 
	HostName ry3.9gpu.com 
	User root 
	Port 21019
```

#图形化界面
- `vncserver: command not found` 解决方案：

```
apt-get update
apt-get install tigervnc-standalone-server tigervnc-common
```
然后设置一下密码就好了（没有设置 view-only passwd）

- 卡在锁屏界面需要输密码
	`vncserver -kill :1`
	- 更新配置禁用锁屏

```
# 1. 停止当前 VNC 进程
vncserver -kill :1

# 2. 直接一键写入新配置（以进入 Xfce 桌面为例）
cat <<EOF > ~/.vnc/xstartup
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
startxfce4 &
EOF

# 3. 赋予执行权限并重新启动
chmod +x ~/.vnc/xstartup
vncserver :1
```


- 黑屏：下载一下桌面应用软件xfce4
	`apt-get update`
	`apt-get install xfce4 xfce4-goodies -y`
	- 更新配置重新打开

```
vncserver -kill :1 cat <<EOF > ~/.vnc/xstartup #!/bin/sh unset SESSION_MANAGER unset DBUS_SESSION_BUS_ADDRESS # 修复部分系统下启动黑屏的问题 [ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup [ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources xsetroot -solid grey # 启动 xfce 核心服务 startxfce4 & EOF chmod +x ~/.vnc/xstartup vncserver :1
```

```
主机运行 vncserver -localhost no :1

本地ssh连接远程主机 ssh -L 5901:127.0.0.1:5901 root@ry3.9gpu.com （输入密码）

Turbo VNC 连接 localhost::5901

密码 {VNCpasswd}
```

![[gleam3.png]]

- 复现过程还有一个 tricky 的事情，权重参数要传入 .zip 路径，如果网盘直接下载是文件夹，自己压缩回去可能会报错。这时候直接下载母文件夹，然后单独复制这个 .zip 就好
## 1.6 重新整理 TODO List 剩余部分 (1.17 Saturday)

- GPU-free：
	- Maniskill3 ✅
	- VGGT ：
		- 下游任务相关工作：LVSM（新视角合成）
		- limitation and improvements 
			- pi^3: 不同于vggt首图世界坐标 ✅
			- MapAnything：任意模态输入 ✅
	- Indoor 3d scene: ReplicaCAD, and AI2THOR , HSSD (https://3dlg-hcvc.github.io/hssd/) 【了解，scale up训练集】
	- GLEAM：
		- 原论文 ✅
		- related works（可以往后放）

- GPU-required：
	- 主机配置（还剩下 ToDesk部分）（等待主机修复）
	- GLEAM 复现 ✅
	- rsl_rl ppo 训练一个标准强化学习任务 dm-control walker-walk (reward curve 上涨到快1000) ✅
	- rsl_rl ppo on gleam-like task
		- input: rgb / depth image
		- output: camera view
		- policy: cnn （重点：处理不定长序列）（ possible solution：max pooling）(vggt)
		- reward: coverage (depth -> point cloud -> voxelize)
		- 理解：cnn 作为 baseline 的方法


## 1.7 Indoor 3D 数据集
### 1.7.1 ReplicaCAD
> repo https://aihabitat.org/datasets/replica_cad/

> ManiSkill3 原生支持 https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/scene_builder/replicacad

- features
	- interactive
	- baked lighting：真实光影信息
	- plenty variations: 不同的家具摆放等


### 1.7.2 Habitat Synthetic Scenes Dataset (HSSD):   An Analysis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal Navigation

> homepage: https://3dlg-hcvc.github.io/hssd/
> repo: https://github.com/3dlg-hcvc/hssd/ (This repository serves as a guide for training and evaluating ObjectNav agents in HSSD, AI2THOR, and HM3D scene datasets using Habitat, and reproducing experiments provided in the HSSD paper.)

not finished


## 1.8 rsl_rl ppo 训练一个标准强化学习任务 dm-control walker-walk
> goal: reward curve 上涨到快1000
> repo: https://github.com/google-deepmind/dm_control
> dm_control tutorial (official): https://colab.research.google.com/github/google-deepmind/dm_control/blob/main/tutorial.ipynb#scrollTo=MpkYHwCqk7W-


> 1.20 讨论更新：重点在于熟悉 rsl_rl 库运行的 ppo pipeline，遂考虑放弃 dm_control ，用更简单的 gymnasium；fs哥提供了一个 demo 脚本 https://pastebin.com/eUfSGJPR  针对 rsl_rl ppo + gym hopper 
> todo：复现脚本结果，拓展 RGB based 代码

### 1.8.1 param - based hopper
- 复现结果：发现训练很不稳定，1000 步平均 reward 最多上涨到 600，又跌回 300 
	- 调整参数 init_noise_std: float = 0.2 （原脚本 1.0） # 初始动作噪声，较低的噪声有助于物理平衡任务初期稳定
	- schedule: str = "adaptive" （原来使用 fixed） # 这里 adaptive 是一个近 penalty 但实际上并不一样的更新策略，具体在本周学习 rsl_rl 库笔记中记录了代码

	- 经过以上两种修改，1000 轮训练耗时 ～29min 效果如下
```
                          Learning iteration 999/1000                            
                            Total steps: 2048000 
                       Steps per second: 1132 
                        Collection time: 0.631s 
                          Learning time: 1.178s 
                        Mean value loss: 193.3120
                    Mean surrogate loss: -0.0051
                      Mean entropy loss: -1.9715
                            Mean reward: 1300.70
                    Mean episode length: 441.78
                  Mean action noise std: 0.13
                         Mean episode r: 192.4580
                        Mean episode _r: 0.1250
                         Mean episode l: 61.8750
                        Mean episode _l: 0.1250
                         Mean episode t: 0.4105
                        Mean episode _t: 0.1250
--------------------------------------------------------------------------------
                         Iteration time: 1.81s
                           Time elapsed: 00:29:18
                                    ETA: 00:00:00
                                    
--- eval.py output 步数 = 518, 总奖励 = 1573.62
```

以上训练效果如下图：
![[hopper-1000steps.mp4]]

### 1.8.2 RGB - based hopper
> 从参数（11 元 vector）观测到 RGB (3 * H * W) 观测

```
<RecordEpisodeStatistics<TimeLimit<OrderEnforcing<PassiveEnvChecker<HopperEnv<Hopper-v4>>>>>> # demo.py 代码中包裹得到的最终 env wrapper
```

- Question：应该先 wrap 图像还是 statistic？
	- 进一步的思考：归根结底 statistic 应该还是基于十一元参数计算得到的（也就是说即使策略变成基于 RGB，由于奖励函数并不是纯粹的来自于 RGB 转化，这种策略近似一种特权观测？）；最理想情况是否是 RGB to reward 也被 well-pretrained？
	- 理解有错，实际上奖励是环境里定义的，而 obs 才是 agent 真正得到的内容，如果 actor or critic 得到除了真实环境能得到的，才称为特权观测。在这个任务里不存在，把奖励传出来就好。

- 遇到的问题与解决：
	- `class CustomActorCritic_CNN(ActorCriticCNN):` 继承后覆盖一些函数，因为注意到原来的 `ActorCriticCNN` 是同时包含 MLP 和 CNN 两个部分，接受 RGB 和 原来的参数两种输入，覆盖直接删掉原来参数处理流程
	- `train_config` 的修改：
```
	"obs_groups": {
		"policy": ["obs"], # 没有 policy_2d 这种东西，AI 在扯淡
		"critic": ["obs"], #
	},
	"policy": {
		"class_name": cfg.policy.class_name,
		"init_noise_std": cfg.policy.init_noise_std,
		"actor_hidden_dims": cfg.policy.actor_hidden_dims,
		"critic_hidden_dims": cfg.policy.critic_hidden_dims,
		"activation": cfg.policy.activation,
		"actor_cnn_cfg": {
			"obs": {
				'kernel_size': [8, 4, 3],
				'output_channels': [32, 64, 64],
				'stride': [4, 2, 1],
			}
		},
		"critic_cnn_cfg": {
			"obs": {
				'kernel_size': [8, 4, 3],
				'output_channels': [32, 64, 64],
				'stride': [4, 2, 1],		
			}
		}
	},
```

- 最终训练过程输出如下：
	- 除了速度比较慢也没什么不好的，八百多轮训到了 500 左右 rewards
	- 写了一个渲染脚本，发现也确实在学了，这个任务差不多（？）

![[train_rgb_hopper.png]]![[hopper_rgb_700steps.mp4]]



### 1.8.3 rsl_rl ppo on gleam-like task
- problem settings
	- input: rgb / depth image
	- output: camera view
	- policy: cnn （重点：处理不定长序列）（ possible solution：max pooling）(vggt)
	- reward: coverage (depth -> point cloud -> voxelize)
	- 理解：cnn 作为 baseline 的方法