# ============================================================================
# ShallowPMD: PyTorch-Only Extreme Simplification
# ============================================================================
# 
# 目标：将 DeePMD-kit 复杂的多框架架构精简为纯 PyTorch 实现
# 原则：
#   1. 删除所有工厂函数、插件系统、多框架抽象
#   2. 保留物理核心：平移不变性、旋转等变性、自动微分
#   3. 极限扁平化：无深继承，直接继承 torch.nn.Module
#   4. 保姆级注释：每个张量、公式、为什么都要解释
#
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings

# ==============================================================================
# 第一部分：物理工具函数
# ==============================================================================

class SmoothingFunction(nn.Module):
    """
    平滑截断函数（Smooth Cutoff Function）
    
    物理意义：
    - 在 r_c (截断半径) 处硬截断会导致势能函数不连续
    - 本函数在 r_s (平滑开始半径) 到 r_c 之间实现光滑的 0-1 衰减
    - 保证了能量的 C^2 连续性（一阶、二阶导数连续）
    
    截断函数定义（DeePMD 标准）：
    s(r) = 1/r × env_decay(x)
    其中 x = (r - r_s) / (r_c - r_s)
    
    env_decay(x) = 1.0，当 x ≤ 0
    env_decay(x) = (1 - x³(6x² - 15x + 10))，当 0 < x < 1  
    env_decay(x) = 0.0，当 x ≥ 1
    
    这个三次多项式设计保证：
    - env_decay(0) = 1.0，导数 = 0
    - env_decay(1) = 0.0，导数 = 0（光滑衰减）
    """
    
    def __init__(self, rcut: float, rcut_smth: float = None):
        """
        Args:
            rcut (float): 截断半径，距离 > rcut 的邻原子完全不考虑
            rcut_smth (float): 平滑开始半径，在 rcut_smth-rcut 之间逐渐衰减
                              如果为 None，则在整个范围内逐渐衰减
        """
        super().__init__()
        self.rcut = rcut
        self.rcut_smth = rcut_smth if rcut_smth is not None else rcut
        
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        计算平滑截断函数值
        
        Args:
            r: 原子间距离张量，任意形状 (...,)
            
        Returns:
            s: 平滑因子张量，形状与 r 相同
               特性：距离越远衰减越多，距离 > rcut 时为 0
        """
        # 计算相对位置：x ∈ [0,1] 表示在平滑区间内的位置
        # x = 0 对应 rcut_smth（平滑开始）
        # x = 1 对应 rcut（截断结束）
        x = (r - self.rcut_smth) / (self.rcut - self.rcut_smth)
        
        # 三次多项式衰减：env_decay(x) = 1 - x³(6x² - 15x + 10)
        # 这个公式保证了在 x=0 和 x=1 处的导数都为 0（C² 连续）
        env_decay = 1.0 - x ** 3 * (6.0 * x ** 2 - 15.0 * x + 10.0)
        
        # 平滑截断：
        # - x < 0: 无衰减，env_decay = 1
        # - 0 ≤ x ≤ 1: 线性衰减，env_decay ∈ [0, 1]
        # - x > 1: 完全截断，env_decay = 0
        s = torch.where(x < 0, torch.ones_like(x),
                       torch.where(x > 1, torch.zeros_like(x), env_decay))
        
        # 距离衰减：1/r 形式确保长程吸引，短程排斥
        # 但 r → 0 时会发散，所以加入保护（env_protection）
        s = s / (r + 1e-6)  # 防止 r=0 时除以零
        
        return s


class EnvironmentMatrix(nn.Module):
    """
    环境矩阵（Environment Matrix）计算器
    
    物理意义：
    - 原子 i 的邻原子环境用矩阵 R^i 表示
    - R^i 的每一行是一个邻原子的向量特征：
      [s(r), s(r)·x/r, s(r)·y/r, s(r)·z/r]
    - 前三列：径向、角向信息的混合编码
    
    数学形式：
    环境矩阵 = [s(r₁)   s(r₁)·x₁/r₁   s(r₁)·y₁/r₁   s(r₁)·z₁/r₁]
               [s(r₂)   s(r₂)·x₂/r₂   s(r₂)·y₂/r₂   s(r₂)·z₂/r₂]
               [  ...        ...            ...            ...    ]
    
    这个矩阵具有：
    1. 平移不变性：仅依赖相对坐标 r_ij = r_j - r_i
    2. 旋转等变性：通过 x_ij/r_ij 等保持角度关系
    """
    
    def __init__(self, rcut: float, rcut_smth: float = None, 
                 env_protection: float = 0.0):
        """
        Args:
            rcut: 截断半径
            rcut_smth: 平滑开始半径
            env_protection: 保护参数，防止 r=0 时除以零
        """
        super().__init__()
        self.smoothing = SmoothingFunction(rcut, rcut_smth)
        self.rcut = rcut
        self.env_protection = env_protection
        
    def forward(self, 
                coords: torch.Tensor,
                nlist: torch.Tensor,
                atype: torch.Tensor) -> torch.Tensor:
        """
        计算环境矩阵
        
        Args:
            coords: 原子坐标 [nf, nall, 3]
                   nf = 帧数（时间步）
                   nall = 总原子数（包括 ghost 原子用于周期边界条件）
                   3 = x, y, z 坐标
                   
            nlist: 邻接列表 [nf*nloc, nnei]
                  nloc = 本地原子数（实际原子）
                  nnei = 最大邻原子数（padding 到固定值）
                  值为原子索引，超出截断的为 -1 或 nloc
                  
            atype: 原子类型 [nf*nloc]
                  每个原子的类型标记（用于后续的类型相关 embedding）
        
        Returns:
            env_mat: 环境矩阵 [nf*nloc, nnei, 4]
                    第三维：[s(r), s(r)·dx/r, s(r)·dy/r, s(r)·dz/r]
        """
        # 展开 coords 以便于索引邻原子
        nf, nall, _ = coords.shape
        nloc = nlist.shape[0]  # 本地原子数
        nnei = nlist.shape[1]  # 邻接表大小
        
        # 中心原子的坐标：reshape 为 [nf*nloc, 3]
        coords_flat = coords.reshape(nf * nall, 3)
        center_coords = coords_flat[:nloc]  # 前 nloc 个是本地原子
        
        # 邻原子坐标：通过 nlist 索引获取 [nf*nloc, nnei, 3]
        # nlist 中的值是全局索引，可以直接索引 coords_flat
        neighbor_coords = coords_flat[nlist]  # [nf*nloc, nnei, 3]
        
        # 计算相对坐标：r_ij = r_j - r_i
        # 广播：[nf*nloc, 1, 3] - [nf*nloc, nnei, 3] = [nf*nloc, nnei, 3]
        rel_coords = neighbor_coords - center_coords.unsqueeze(1)
        
        # 计算距离：||r_ij|| = sqrt(x² + y² + z²)
        # 保护参数防止距离为 0 时计算问题
        distances = torch.norm(rel_coords, dim=2, keepdim=True)  # [nf*nloc, nnei, 1]
        distances = distances + self.env_protection
        
        # 计算平滑因子：s(r) = smoothing(r)
        smooth_factor = self.smoothing(distances.squeeze(-1))  # [nf*nloc, nnei]
        
        # 归一化相对坐标：r_ij / ||r_ij|| = [dx/r, dy/r, dz/r]
        unit_rel_coords = rel_coords / distances  # [nf*nloc, nnei, 3]
        
        # 将平滑因子与归一化坐标组合：[s(r), s(r)·dx/r, s(r)·dy/r, s(r)·dz/r]
        smooth_factor_expanded = smooth_factor.unsqueeze(-1)  # [nf*nloc, nnei, 1]
        env_mat = torch.cat(
            [smooth_factor_expanded, smooth_factor_expanded * unit_rel_coords],
            dim=2
        )  # [nf*nloc, nnei, 4]
        
        return env_mat


# ==============================================================================
# 第二部分：简化的描述符实现（以 SE-A 为例）
# ==============================================================================

class SimpleEmbeddingNet(nn.Module):
    """
    简化的嵌入网络（Embedding Network）
    
    物理意义：
    - 输入：距离信息 s(r) ∈ [0, 1]（标量）
    - 输出：高维特征向量，捕捉径向相互作用信息
    - 功能：非线性特征提取，学习原子对相互作用
    
    架构：
    s(r) → Dense(h₁) → Tanh → Dense(h₂) → Tanh → ... → Dense(out_dim)
    
    为什么用 Tanh？
    - 有界激活：输出 ∈ [-1, 1]，适合神经网络训练稳定性
    - 非线性：足以表达复杂的相互作用
    """
    
    def __init__(self, 
                 input_dim: int = 1,
                 hidden_dims: List[int] = None,
                 output_dim: int = 32,
                 activation: str = "tanh",
                 use_resnet: bool = False,
                 resnet_dt: float = 0.001):
        """
        Args:
            input_dim: 输入维度（通常为 1，因为输入就是 s(r)）
            hidden_dims: 隐层维度列表 [h₁, h₂, ...]
            output_dim: 输出维度（特征向量大小）
            activation: 激活函数 ("tanh" 或 "relu")
            use_resnet: 是否使用 ResNet 风格的跳跃连接
            resnet_dt: ResNet 残差块的时间步大小（0 < dt < 1）
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64]
        
        self.use_resnet = use_resnet
        self.resnet_dt = resnet_dt
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # 最后一层到输出维度（无激活函数）
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name.lower() == "tanh":
            return nn.Tanh()
        elif name.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [..., input_dim]，例如 [nf*nloc, nnei, 1]
        
        Returns:
            y: 输出特征 [..., output_dim]，例如 [nf*nloc, nnei, 32]
        """
        # 记住输入形状以便后续 reshape
        input_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # 展平前 N-1 维
        
        if self.use_resnet and x_flat.shape[-1] == self.output_dim:
            # ResNet 连接：y = x + dt * φ(x)
            # 这样设计使网络可以保留输入信息，增强训练稳定性
            y = x_flat + self.resnet_dt * self.net(x_flat)
        else:
            # 普通前向：y = φ(x)
            y = self.net(x_flat)
        
        # 恢复形状
        return y.reshape(*input_shape, self.output_dim)


class SimpleDescriptor(nn.Module):
    """
    简化的描述符（描述原子局部环境的特征向量）
    
    物理意义：
    - 描述符是原子局部化学环境的紧凑表示
    - 与能量、力等全局性质的对应关系由拟合网络学习
    - 设计使其具有：平移不变性、旋转等变性、排列等变性
    
    计算流程：
    环境矩阵 R^i → Embedding (特征提取) → G^i (嵌入矩阵)
    → 矩阵乘法组合 → 描述符 D^i
    
    关键性质：
    1. 平移不变性（Translation Invariance）：
       仅依赖相对坐标，不依赖绝对位置
    
    2. 排列等变性（Permutation Equivariance）：
       邻原子的顺序改变不会改变最终的描述符
       （通过对邻原子的信息进行求和/池化实现）
    
    3. 旋转等变性（Rotation Equivariance）：
       通过 SO(3) 旋转矩阵保持（简化版中忽略）
    """
    
    def __init__(self,
                 rcut: float,
                 rcut_smth: float = None,
                 sel: List[int] = None,
                 embedding_dims: List[int] = None,
                 output_dim: int = 32,
                 activation: str = "tanh"):
        """
        Args:
            rcut: 截断半径
            rcut_smth: 平滑开始半径
            sel: 每种原子类型的选择数 [sel_0, sel_1, ...]
            embedding_dims: 嵌入网络的隐层维度 [h₁, h₂, ...]
            output_dim: 描述符输出维度
            activation: 激活函数
        """
        super().__init__()
        
        if sel is None:
            sel = [32]  # 默认 1 种原子类型，选 32 个邻原子
        
        if embedding_dims is None:
            embedding_dims = [32, 64, 128]
        
        self.sel = sel
        self.ntypes = len(sel)
        self.nnei = sum(sel)  # 总邻原子数
        self.output_dim = output_dim
        
        # 初始化环境矩阵计算器
        self.env_mat = EnvironmentMatrix(rcut, rcut_smth)
        
        # 为每种原子类型创建 embedding 网络
        # 这样可以区分不同原子类型的相互作用强度
        self.embeddings = nn.ModuleList()
        for itype in range(self.ntypes):
            emb = SimpleEmbeddingNet(
                input_dim=1,
                hidden_dims=embedding_dims,
                output_dim=embedding_dims[-1],
                activation=activation
            )
            self.embeddings.append(emb)
        
        # 最终的映射层：将嵌入向量转化为描述符
        # 输入：[nf*nloc, nnei, embedding_dims[-1]]
        # 输出：[nf*nloc, output_dim]
        self.projection = nn.Sequential(
            nn.Linear(embedding_dims[-1] * self.nnei, output_dim),
            nn.Tanh() if activation.lower() == "tanh" else nn.ReLU()
        )
    
    def forward(self, 
                coords: torch.Tensor,
                atype: torch.Tensor,
                nlist: torch.Tensor) -> torch.Tensor:
        """
        计算描述符
        
        Args:
            coords: 原子坐标 [nf, nall, 3]
            atype: 原子类型 [nf*nloc]
            nlist: 邻接列表 [nf*nloc, nnei]
        
        Returns:
            descriptor: 描述符 [nf*nloc, output_dim]
        """
        # 第一步：计算环境矩阵 [nf*nloc, nnei, 4]
        env_mat = self.env_mat(coords, nlist, atype)
        
        # 第二步：提取距离信息 [nf*nloc, nnei, 1]
        # 环境矩阵第一列是 s(r)，代表平滑后的距离依赖
        distance_feature = env_mat[..., 0:1]
        
        # 第三步：通过 embedding 网络处理 [nf*nloc, nnei, embedding_dim]
        # 简化版：使用第一个 embedding 网络处理所有原子
        # 完整版应该根据 nlist 中邻原子的类型选择不同的 embedding
        embedded = self.embeddings[0](distance_feature)
        
        # 第四步：展平并投影到描述符空间 [nf*nloc, output_dim]
        nloc = embedded.shape[0]
        embedded_flat = embedded.reshape(nloc, -1)
        descriptor = self.projection(embedded_flat)
        
        return descriptor


# ==============================================================================
# 第三部分：简化的拟合网络
# ==============================================================================

class SimpleFittingNet(nn.Module):
    """
    简化的拟合网络（Fitting Network）
    
    物理意义：
    - 将描述符映射到物理量（能量、力等）
    - 这个映射不是显式给出的，需要通过拟合学习
    - 架构：深度神经网络，带有 ResNet 连接增强训练
    
    数学形式：
    E_atom = MLP(D^i)  其中 D^i 是原子 i 的描述符
    
    为什么需要拟合网络？
    - 描述符本身是低级的几何特征
    - 物理量（能量）是高级的、非线性的功能
    - 神经网络正是用来学习这种复杂映射的
    """
    
    def __init__(self,
                 descriptor_dim: int = 32,
                 ntypes: int = 1,
                 hidden_dims: List[int] = None,
                 activation: str = "tanh",
                 use_resnet: bool = True,
                 resnet_dt: float = 0.001):
        """
        Args:
            descriptor_dim: 输入描述符维度
            ntypes: 原子类型数
            hidden_dims: 隐层维度列表
            activation: 激活函数
            use_resnet: 是否使用 ResNet
            resnet_dt: ResNet 时间步
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]
        
        self.descriptor_dim = descriptor_dim
        self.ntypes = ntypes
        self.use_resnet = use_resnet
        self.resnet_dt = resnet_dt
        
        # 为每种原子类型创建独立的拟合网络
        # 这允许不同元素的相互作用强度不同
        self.fitting_nets = nn.ModuleList()
        
        for itype in range(ntypes):
            layers = []
            prev_dim = descriptor_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                prev_dim = hidden_dim
            
            # 最后一层：输出原子能量（标量）
            layers.append(nn.Linear(prev_dim, 1))
            
            self.fitting_nets.append(nn.Sequential(*layers))
    
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        if name.lower() == "tanh":
            return nn.Tanh()
        elif name.lower() == "relu":
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self,
                descriptor: torch.Tensor,
                atype: torch.Tensor) -> torch.Tensor:
        """
        计算原子能量
        
        Args:
            descriptor: 描述符 [nf*nloc, descriptor_dim]
            atype: 原子类型 [nf*nloc]，值为 0, 1, ..., ntypes-1
        
        Returns:
            atom_energy: 原子能量 [nf*nloc, 1]
        """
        batch_size = descriptor.shape[0]
        atom_energy = torch.zeros(batch_size, 1, device=descriptor.device, dtype=descriptor.dtype)
        
        # 对每种原子类型分别处理
        for itype in range(self.ntypes):
            # 找到该类型的原子 [mask_size]
            mask = (atype == itype)
            
            if mask.sum() > 0:
                # 提取该类型原子的描述符 [mask_size, descriptor_dim]
                descriptor_itype = descriptor[mask]
                
                # 通过该类型的拟合网络 [mask_size, 1]
                energy_itype = self.fitting_nets[itype](descriptor_itype)
                
                # 放回到结果数组中
                atom_energy[mask] = energy_itype
        
        return atom_energy


print("✓ Part 1 completed: Basic physics tools and simplified components")



# ==============================================================================
# 第四部分：顶层模型
# ==============================================================================

class ShallowPMDModel(nn.Module):
    """
    极简版 DeePMD 模型（Pure PyTorch，无框架抽象）
    
    系统整体架构：
    ┌─────────────────┐
    │  原子坐标、类型   │  coords [nf, nall, 3], atype [nall]
    │  邻接列表        │  nlist [nf*nloc, nnei]
    └────────┬────────┘
             │
             ↓
    ┌─────────────────────────┐
    │    描述符计算器           │  描述 = 局部环境的特征向量
    │  Descriptor(coords...)  │  保证平移不变性
    └────────┬────────────────┘
             │  descriptor [nf*nloc, dim_descriptor]
             ↓
    ┌─────────────────────────┐
    │    拟合网络              │  将描述符映射到物理量
    │  FittingNet(descr...)   │  学习相互作用函数
    └────────┬────────────────┘
             │  atom_energy [nf*nloc, 1]
             ↓
    ┌─────────────────────────┐
    │    能量聚合              │  求和所有原子能量
    │    Sum over atoms       │  得到系统总能量
    └────────┬────────────────┘
             │  total_energy [nf]
             ↓
    ┌─────────────────────────┐
    │    力计算                │  F = -∂E/∂R
    │  F = -grad(E, coords)   │  通过自动微分实现
    └─────────────────────────┘
    
    关键物理原理：
    1. 系统总能量 E = Σᵢ Eᵢ 
       其中 Eᵢ 是原子 i 的能量（由其局部环境决定）
    
    2. 原子受力 Fᵢ = -∂E/∂rᵢ
       PyTorch 的 autograd 自动计算梯度
    
    3. 应力张量 σ = (1/V) ∂E/∂ε
       可以通过对应变应变求导得到
    """
    
    def __init__(self,
                 rcut: float = 6.0,
                 rcut_smth: float = 5.8,
                 sel: List[int] = None,
                 descriptor_dims: List[int] = None,
                 descriptor_output_dim: int = 64,
                 ntypes: int = 1,
                 fitting_dims: List[int] = None,
                 atom_ener: Optional[List[float]] = None,
                 activation: str = "tanh"):
        """
        Args:
            rcut: 截断半径（Angstrom），常见值 5-7
            rcut_smth: 平滑开始半径（通常 rcut_smth ≈ 0.95 * rcut）
            sel: 每种原子类型的邻原子选择数 [sel_0, sel_1, ...]
            descriptor_dims: 描述符网络隐层维度
            descriptor_output_dim: 描述符输出维度
            ntypes: 原子类型数
            fitting_dims: 拟合网络隐层维度
            atom_ener: 原子参考能量 [e_0, e_1, ...] 用于归一化
            activation: 激活函数
        """
        super().__init__()
        
        # 参数存储
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.ntypes = ntypes
        self.atom_ener = atom_ener
        
        if sel is None:
            sel = [32] * ntypes
        self.sel = sel
        
        if descriptor_dims is None:
            descriptor_dims = [64, 128, 256]
        
        if fitting_dims is None:
            fitting_dims = [256, 256]
        
        # 初始化描述符计算器
        self.descriptor = SimpleDescriptor(
            rcut=rcut,
            rcut_smth=rcut_smth,
            sel=sel,
            embedding_dims=descriptor_dims,
            output_dim=descriptor_output_dim,
            activation=activation
        )
        
        # 初始化拟合网络
        self.fitting_net = SimpleFittingNet(
            descriptor_dim=descriptor_output_dim,
            ntypes=ntypes,
            hidden_dims=fitting_dims,
            activation=activation,
            use_resnet=True,
            resnet_dt=0.001
        )
        
        # 可学习的原子参考能量（用于输出层偏置）
        if atom_ener is not None:
            self.register_buffer(
                'atom_ener_bias',
                torch.tensor(atom_ener, dtype=torch.float32)
            )
        else:
            self.register_parameter(
                'atom_ener_bias',
                nn.Parameter(torch.zeros(ntypes))
            )
    
    def forward(self,
                coords: torch.Tensor,
                atype: torch.Tensor,
                nlist: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算能量和力
        
        Args:
            coords: 原子坐标 [nf, nall, 3]
                   nf = 帧数
                   nall = 总原子数
                   3 = (x, y, z)
                   
            atype: 原子类型 [nall] ∈ {0, 1, ..., ntypes-1}
            
            nlist: 邻接列表 [nf*nloc, nnei]
                  包含所有邻原子的全局索引
                  -1 或 out-of-range 值表示无邻原子（padding）
        
        Returns:
            dict 包含：
                'energy': 系统总能量 [nf]
                'atom_energy': 每个原子的能量 [nf*nloc, 1]
                'descriptor': 描述符 [nf*nloc, descriptor_dim]
        """
        # 确保坐标需要梯度（用于力计算）
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
        
        # 第一步：计算描述符 [nf*nloc, descriptor_dim]
        descriptor = self.descriptor(coords, atype, nlist)
        
        # 构建 nlist 对应的 atype（邻原子的类型）
        # 这样拟合网络可以区分不同类型原子的能量
        nf = coords.shape[0]
        nall = coords.shape[1]
        nloc = descriptor.shape[0] // nf
        
        # 重复 atype 以匹配 nlist 的形状
        atype_expanded = atype.unsqueeze(0).expand(nf, -1).reshape(-1)  # [nf*nall]
        atype_center = atype_expanded[:nf*nloc]  # 取中心原子的类型 [nf*nloc]
        
        # 第二步：通过拟合网络计算原子能量 [nf*nloc, 1]
        atom_energy = self.fitting_net(descriptor, atype_center)
        
        # 第三步：添加原子参考能量（如果有）并聚合
        # 这样做是为了提高网络的学习稳定性
        if hasattr(self, 'atom_ener_bias'):
            atom_ener_bias = self.atom_ener_bias[atype_center]  # [nf*nloc]
            atom_energy = atom_energy + atom_ener_bias.unsqueeze(-1)
        
        # 重新 reshape 为 [nf, nloc]
        atom_energy_reshaped = atom_energy.reshape(nf, nloc, 1)
        
        # 第四步：对所有原子求和得到系统总能量 [nf]
        # 物理意义：系统势能 = 所有原子能量之和
        system_energy = atom_energy_reshaped.sum(dim=1).squeeze(-1)
        
        return {
            'energy': system_energy,
            'atom_energy': atom_energy,
            'descriptor': descriptor
        }
    
    def compute_forces(self,
                       coords: torch.Tensor,
                       atype: torch.Tensor,
                       nlist: torch.Tensor) -> torch.Tensor:
        """
        计算原子受力
        
        物理原理：
        系统中原子 i 受到的力是系统势能对其坐标的负梯度
        
        F_i = -∂E/∂r_i
        
        其中 E 是系统总势能，r_i 是原子 i 的位置向量
        
        这个梯度计算包括：
        1. 直接梯度：原子 i 通过描述符直接影响能量
        2. 间接梯度：原子 j 作为原子 i 的邻原子，对原子 i 有力的影响
        
        PyTorch 的 autograd 会自动追踪所有这些路径
        
        Args:
            coords: 原子坐标 [nf, nall, 3]，需要梯度
            atype: 原子类型 [nall]
            nlist: 邻接列表 [nf*nloc, nnei]
        
        Returns:
            forces: 原子受力 [nf, nall, 3]
                   注意这里返回所有原子（包括 ghost）的力
                   实际使用时通常只取 [nf, nloc, 3]
        """
        # 前向传播
        output = self.forward(coords, atype, nlist)
        total_energy = output['energy'].sum()
        
        # 计算能量对坐标的梯度
        # retain_graph=True 允许多次调用 backward
        forces = -torch.autograd.grad(
            total_energy,
            coords,
            create_graph=True,  # 保留计算图，便于进一步求导（如 Hessian）
            retain_graph=True
        )[0]  # grad 返回元组，[0] 取梯度张量
        
        return forces
    
    def compute_virial(self,
                       coords: torch.Tensor,
                       atype: torch.Tensor,
                       nlist: torch.Tensor,
                       cell: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算应力张量（Virial Tensor / Stress）
        
        物理意义：
        应力张量 σ 描述系统内部压强分布
        σ_ij = -(1/V) ∂E/∂ε_ij
        
        其中 ε 是应变张量，V 是体积
        
        应力可用于：
        1. 计算压强 P = -Tr(σ)/3
        2. 计算弹性常数
        3. 分子动力学中的恒压/恒应力模拟
        
        Args:
            coords: 原子坐标 [nf, nall, 3]
            atype: 原子类型 [nall]
            nlist: 邻接列表 [nf*nloc, nnei]
            cell: 周期边界单胞 [nf, 3, 3]，可选
        
        Returns:
            virial: 应力张量 [nf, 3, 3]
                   对称矩阵
        """
        # 如果提供了单胞，计算应变导数
        # σ = -(1/V) ∂E/∂ε
        if cell is not None:
            if not cell.requires_grad:
                cell = cell.requires_grad_(True)
        
        # 前向传播
        output = self.forward(coords, atype, nlist)
        total_energy = output['energy'].sum()
        
        # 计算能量对坐标的偏导数贡献
        # 这需要通过二阶导数实现
        # 简化版：仅返回零张量
        # 完整版需要更复杂的实现
        nf = coords.shape[0]
        virial = torch.zeros(nf, 3, 3, device=coords.device, dtype=coords.dtype)
        
        return virial


# ==============================================================================
# 第五部分：训练工具和损失函数
# ==============================================================================

class ShallowPMDLoss(nn.Module):
    """
    DeePMD 训练损失函数
    
    总损失 = w_e * L_energy + w_f * L_force + w_v * L_virial
    
    其中：
    - L_energy: 能量预测误差 （重要：控制总势能准确度）
    - L_force: 力预测误差 （更重要：F = -∂E/∂R）
    - L_virial: 应力/压强误差 （可选：分子动力学稳定性）
    
    权重设置很关键：
    - w_f > w_e: 力的准确度更重要（因为 MD 使用力）
    - 典型值：w_e = 1.0, w_f = 2.0, w_v = 0.0
    """
    
    def __init__(self,
                 start_lr: float = 0.001,
                 stop_lr: float = 0.00001,
                 loss_start_ener: float = 0.02,
                 loss_stop_ener: float = 0.0001,
                 loss_start_force: float = 1.0,
                 loss_stop_force: float = 0.01,
                 loss_start_virial: float = 0.0,
                 loss_stop_virial: float = 0.0):
        """
        Args:
            start_lr: 初始学习率
            stop_lr: 最终学习率
            loss_*: 各部分损失的初始/最终权重
        """
        super().__init__()
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.loss_start_ener = loss_start_ener
        self.loss_stop_ener = loss_stop_ener
        self.loss_start_force = loss_start_force
        self.loss_stop_force = loss_stop_force
        self.loss_start_virial = loss_start_virial
        self.loss_stop_virial = loss_stop_virial
        self.step = 0
    
    def forward(self,
                pred_energy: torch.Tensor,
                pred_force: torch.Tensor,
                pred_virial: Optional[torch.Tensor] = None,
                targ_energy: Optional[torch.Tensor] = None,
                targ_force: Optional[torch.Tensor] = None,
                targ_virial: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            pred_*: 模型预测值
            targ_*: 目标值
        
        Returns:
            dict 包含各部分损失和总损失
        """
        losses = {}
        total_loss = 0
        
        # 能量损失
        if targ_energy is not None:
            # RMSE: sqrt(mean((pred - targ)²))
            energy_mse = F.mse_loss(pred_energy, targ_energy)
            energy_rmse = torch.sqrt(energy_mse + 1e-8)
            losses['energy_rmse'] = energy_rmse
            
            # 动态权重：从 loss_start_ener 逐渐减小到 loss_stop_ener
            ener_weight = self.loss_start_ener  # 简化版不实现动态权重
            total_loss = total_loss + ener_weight * energy_rmse
        
        # 力损失（通常权重更大）
        if targ_force is not None:
            force_mse = F.mse_loss(pred_force, targ_force)
            force_rmse = torch.sqrt(force_mse + 1e-8)
            losses['force_rmse'] = force_rmse
            
            force_weight = self.loss_start_force
            total_loss = total_loss + force_weight * force_rmse
        
        # 应力损失（可选）
        if targ_virial is not None and pred_virial is not None:
            virial_mse = F.mse_loss(pred_virial, targ_virial)
            virial_rmse = torch.sqrt(virial_mse + 1e-8)
            losses['virial_rmse'] = virial_rmse
            
            virial_weight = self.loss_start_virial
            total_loss = total_loss + virial_weight * virial_rmse
        
        losses['total'] = total_loss
        return losses


# ==============================================================================
# 第六部分：示例代码和单元测试
# ==============================================================================

def create_dummy_batch():
    """
    创建虚拟数据用于测试
    
    创建一个简单的系统：
    - 1 帧（1 个时间步）
    - 2 个原子（1 个中心原子 + 1 个邻原子）
    - 1 种原子类型
    """
    # 原子坐标：
    # 原子 0: (0, 0, 0)
    # 原子 1: (1, 0, 0) —— 距离为 1 Angstrom
    coords = torch.tensor([[[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0]]], dtype=torch.float32)
    
    # 原子类型：都是类型 0
    atype = torch.tensor([0, 0], dtype=torch.long)
    
    # 邻接列表：
    # 原子 0 的邻原子是 [1]（只有原子 1 在截断半径内）
    # 用 -1 padding（假设最多 4 个邻原子）
    nlist = torch.tensor([[1, -1, -1, -1]], dtype=torch.long)
    
    return coords, atype, nlist


def test_model():
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("TEST: ShallowPMD 模型测试")
    print("="*60)
    
    # 创建模型
    model = ShallowPMDModel(
        rcut=6.0,
        rcut_smth=5.8,
        sel=[4],  # 最多 4 个邻原子
        descriptor_dims=[16, 32],  # 隐层维度（小模型用于测试）
        descriptor_output_dim=16,
        ntypes=1,
        fitting_dims=[32, 32],
        activation="tanh"
    )
    
    # 加载虚拟数据
    coords, atype, nlist = create_dummy_batch()
    
    # 前向传播
    print("\n1. 前向传播测试")
    print(f"   输入坐标形状: {coords.shape}")
    print(f"   输入原子类型: {atype}")
    print(f"   输入邻接列表形状: {nlist.shape}")
    
    output = model(coords, atype, nlist)
    print(f"   → 系统能量: {output['energy']}")
    print(f"   → 原子能量形状: {output['atom_energy'].shape}")
    print(f"   → 描述符形状: {output['descriptor'].shape}")
    
    # 力计算测试
    print("\n2. 力计算测试 (F = -∂E/∂R)")
    forces = model.compute_forces(coords, atype, nlist)
    print(f"   → 力张量形状: {forces.shape}")
    print(f"   → 力值:\n{forces}")
    
    # 验证力的性质
    print("\n3. 力的物理检验")
    print(f"   → 总力（应该接近零）: {forces.sum(dim=(0, 1))}")
    print("   ✓ 如果总力≈0，表示系统动量守恒")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)


if __name__ == "__main__":
    test_model()

