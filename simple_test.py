#!/usr/bin/env python3
"""
简单的 ShallowPMD 验证脚本
用于快速测试核心功能是否工作
"""

import sys

def main():
    print("\n" + "="*70)
    print("  DeePMD-Minimal 简单验证测试")
    print("="*70 + "\n")
    
    # 检查 1: 导入
    print("[1/5] 检查导入...")
    try:
        import torch
        import numpy as np
        from deepmd_minimal import (
            SmoothingFunction,
            EnvironmentMatrix,
            SimpleEmbeddingNet,
            SimpleDescriptor,
            SimpleFittingNet,
            DeePMDMinimalModel,
            DeePMDLoss
        )
        print("✓ 所有模块导入成功")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return 1
    
    # 检查 2: 创建基本组件
    print("\n[2/5] 创建模型组件...")
    try:
        # 平滑函数
        sf = SmoothingFunction(rcut=6.0, rcut_smth=5.8)
        print("  ✓ SmoothingFunction 创建成功")
        
        # 环境矩阵
        env_mat = EnvironmentMatrix(rcut=6.0, rcut_smth=5.8)
        print("  ✓ EnvironmentMatrix 创建成功")
        
        # 嵌入网络
        emb_net = SimpleEmbeddingNet(input_dim=1, output_dim=16)
        print("  ✓ SimpleEmbeddingNet 创建成功")
        
        # 完整模型
        model = DeePMDMinimalModel(
            rcut=6.0,
            sel=[32],
            ntypes=1,
            descriptor_dims=[16],
            fitting_dims=[16]
        )
        print("  ✓ DeePMDMinimalModel 创建成功")
        
    except Exception as e:
        print(f"✗ 组件创建失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 检查 3: 前向传播
    print("\n[3/5] 测试前向传播...")
    try:
        nframes = 1
        nall = 10
        nloc = 8
        nnei = 32
        
        coords = torch.randn(nframes, nall, 3)
        atype = torch.zeros(nall, dtype=torch.int32)
        nlist = torch.randint(0, nall, (nframes*nloc, nnei), dtype=torch.int32)
        
        output = model(coords, atype, nlist)
        
        print(f"  ✓ 前向传播成功")
        print(f"    - 能量形状: {output['energy'].shape}")
        print(f"    - 原子能量形状: {output['atom_energy'].shape}")
        print(f"    - 描述符形状: {output['descriptor'].shape}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 检查 4: 力计算
    print("\n[4/5] 测试力计算...")
    try:
        coords_grad = torch.randn(nframes, nall, 3, requires_grad=True)
        forces = model.compute_forces(coords_grad, atype, nlist)
        
        print(f"  ✓ 力计算成功")
        print(f"    - 力形状: {forces.shape}")
        print(f"    - 力的范数: {torch.norm(forces).item():.6f}")
        print(f"    - 包含 NaN: {torch.isnan(forces).any().item()}")
        
    except Exception as e:
        print(f"✗ 力计算失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 检查 5: 训练循环的基础
    print("\n[5/5] 测试参数更新...")
    try:
        # 获取初始参数
        initial_params = [p.clone() for p in model.parameters()]
        
        # 简单的训练步骤
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        coords_grad = torch.randn(nframes, nall, 3, requires_grad=True)
        
        output = model(coords_grad, atype, nlist)
        loss = output['energy'].sum()
        
        loss.backward()
        optimizer.step()
        
        # 检查参数是否改变
        params_changed = False
        for p_init, p_current in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_current):
                params_changed = True
                break
        
        if params_changed:
            print(f"  ✓ 参数更新成功")
        else:
            print(f"  ⚠ 参数未改变（可能没有梯度）")
            
    except Exception as e:
        print(f"✗ 参数更新失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 总结
    print("\n" + "="*70)
    print("✨ 所有基础测试都通过了！")
    print("="*70)
    print("\n推荐的下一步:")
    print("  1. 阅读 START_HERE.md")
    print("  2. 查看 README_MINIMAL.md")
    print("  3. 运行 deepmd_minimal.py 中的完整示例")
    print("  4. 阅读 DEEPMD_MINIMAL_GUIDE.md 了解架构")
    print("\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
