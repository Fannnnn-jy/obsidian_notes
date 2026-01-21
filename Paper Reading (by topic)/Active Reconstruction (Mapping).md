
下表作为本笔记统计内容

| Year      | Title                                                                                       | Path                       |                        |     |
| --------- | ------------------------------------------------------------------------------------------- | -------------------------- | ---------------------- | --- |
|           | GLEAM                                                                                       |                            |                        |     |
| 2024 ECCV | FisherRF: Active View Selection and Mapping with Radiance Fields using Fisher Information   | Information gain / NeRF    | active mapping         | ✅   |
| 2023 ICRA | Uncertainty Guided Policy for Active Robotic 3D Reconstruction using Neural Radiance Fields | Entropy / NeRF             | active recon           | ✅   |
| 2025 RSS  | GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction        | Mutual Info / GS splatting | active recon           |     |
| 2025      | 3R-GS: Best Practice in Optimizing Camera Poses Along with 3DGS                             |                            | camera re-localization |     |
| 2025 ICLR | GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting                          |                            | camera re-localization |     |

## FisherRF: Active View Selection and Mapping with Radiance Fields using Fisher Information

- Method
	- 对于一个参数 w 的 NeRF 网络 $f(x; w)$，$$-\log p(y|x, w) \propto (y - f(x, w))^T (y - f(x, w))$$$$I(w) = - \mathbb{E}_{y \sim p(y|x,w)} \left[ \frac{\partial^2 \log p(y|x, w)}{\partial w^2} | w \right] = \mathbb{H}'' [y | x, w] $$
	- 好吧其实剩下的我也没太看懂，反正就是对于当前步骤我可以直接通过之前的信息和候选的 x 来算一个近似的信息增益（可以不需要知道y，也就是不用渲染就能近似），argmax 就可以取到近似信息增益最大的

## GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting

![[GS-CPR.png]]


## Uncertainty Guided Policy for Active Robotic 3D Reconstruction using Neural Radiance Fields

- Core Innovation：ray-based volumetric uncertainty estimator
	- 在 NeRF 标准设定里，我们用下式来表示一条射线的颜色$$\hat{\textbf{C}}(\textbf{r}) = \sum_{i=1}^{n} w_i \textbf{c}_i$$其中，$w_i​ = T_i​(1−exp(−\sigma_i​\delta_i​)) = \alpha_i \times T_i$ 表示特定 3D 点对投影点颜色的贡献。$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$  代表光线在经过距离 $\delta_i$ 时，被点 $i$ 的物质“挡住”或“捕获”的概率；
	 $T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j \delta_j)$，代表光线从相机出发，穿过前 $i-1$ 个点后，依然能存活（没被挡住）到达第 $i$ 个点的概率。

	- 在传统的 NeRF 中，我们只关心最终合成的颜色 $\hat{\textbf{C}}(\textbf{r})$。但在主动建图（Active Mapping）中，我们更关心这些权重 $w_i$ 的分布形态，因为 $w_i$ 本质上是光线在空间中被阻挡的概率密度函数（PDF）。
	- 作者认为： $w_i$ 的熵 $H(w) = - \sum_{i=1}^n w_i \cdot log \: w_i$ 可以直接反应当前重建的不确定性

## GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction

![[GauSS-MI1.png]]

- 一个特征：基于重建精确度
	- 看了一遍，有一个笼统的整体理解：什么是最好的 3D 重建方向？之前的工作总是觉得，如果这张图带有的真实信息是以前猜不出来的、熵最大的 etc （Q： 如果是直接看背面，完全什么都不知道呢？）。这篇文章给出的视角是：降低原来重建情况不确定性最多的（max mutual info）。
	- 所以 Gauss-MI 的主要思想如下：
		- 由于 GS splatting 的一些数学性质，可以定义一个随机变量代表一个高斯球的可信程度，最开始的时候所有高斯球可信程度都是 0.5，由重建所用的每一张图来更新
		- 我们可以计算出一个值：给定新视角的 obs $Z_k$， 对每一个高斯球的可信度建模如下$$P(r^{[i]}|Z_k) = \dfrac{1}{(\lambda_kL_k)^{\lambda_T T^{[i]}}+1}$$$$L_k = \lambda_c||C-\hat{C}_k|| + (1 - \lambda_c)||D - \hat{D}_k||$$实践中会简化计算，改优化 log odd：$$l^{[i]}_{1:k} = -\lambda_T T^{[i]}log \: \lambda_L L_k + l^{[i]}_{1:k-1}$$
		- 可以如下图算法优化![[GauSS-MI2.png]]

		- 根据前面提到的思想，作者目标则是衡量 **GS splatting 的先验熵** 与 视角 $k$ obs $Z_k$ 提供后的 **条件熵** 之差，也就是二者 MI (Mutual Information)。算法如下：![[GauSS-MI3.png]]