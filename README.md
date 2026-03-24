# 🚀 DeepSpeed-Inference

> 基于 Tiny-DeepSpeed 的增强版实现，聚焦大模型分布式训练优化与显存受限场景下的高效推理。

---

## 📌 项目简介

本项目基于原始 Tiny-DeepSpeed 进行二次开发与系统性优化，围绕 **DeepSpeed 核心并行机制（ZeRO1/2/3）**、**大模型推理加速** 以及 **资源受限环境下的性能优化** 展开。

在原有“教学级实现”的基础上，进一步增强为**具备工程实践价值的轻量级分布式训练与推理框架**，适用于：

* 大模型训练机制理解与验证
* 显存优化策略研究（ZeRO / Offloading）
* 单卡 / 小规模集群环境下的推理优化实践

---

## ⚙️ 核心改进（相较原版）

### ✅ 1. 分布式训练能力增强

* 完整实现并验证 **ZeRO Stage 1 / 2 / 3**
* 对比 DDP 与 ZeRO 系列策略的显存与性能差异
* 支持梯度分片与参数分布式管理

### ✅ 2. 显存优化与大模型支持

* 基于 **Meta Device Initialization** 优化模型初始化流程
* 引入参数按需加载机制，降低初始化与训练显存占用
* 支持更大规模模型在有限 GPU（如 24GB VRAM）环境下运行

### ✅ 3. 推理性能优化（重点增强）

* 实现 **CPU-GPU 异构推理（Offloading）**
* 支持参数分片加载，减少 GPU 常驻显存占用
* 优化推理路径，提升吞吐与降低延迟

### ✅ 4. 计算与通信优化

* 实现 **Compute-Communication Overlap**
* 优化梯度同步策略，减少通信阻塞
* 提升多卡训练效率

### ✅ 5. 工程化改进

* 重构部分模块结构，提高代码可读性与扩展性
* 增强训练稳定性（异常处理 / 同步机制优化）
* 提供统一实验接口，便于对比不同并行策略

---

## 📊 性能对比（GPT-2）

|     Methods     | 1 GPU | DDP - 2 GPU | Zero1 - 2 GPU | Zero2 - 2 GPU | Zero3 - 2 GPU |
| :-------------: | :---: | :---------: | :-----------: | :-----------: | :-----------: |
|  **GPT2-small** |  4.65 |     4.75    |      4.08     |      3.79     |      3.69     |
| **GPT2-medium** | 10.12 |    10.23    |      8.65     |      8.25     |      7.73     |
|  **GPT2-large** | 17.35 |    17.46    |     14.08     |     12.89     |     11.01     |

👉 在实际实验中，ZeRO-3 相比 DDP 可降低约 **30%~40% 显存占用**

---

## 🧪 快速开始

### 环境依赖

* Python 3.11
* PyTorch 2.3.1 (CUDA)
* Triton 2.3.1

### 安装

```bash
git clone https://github.com/liangyuwang/Tiny-DeepSpeed.git
cd Tiny-DeepSpeed
```

---

## ▶️ 运行示例

```bash
# 单卡训练
python example/single_device/train.py

# DDP
torchrun --nproc_per_node=2 example/ddp/train.py

# ZeRO-1
torchrun --nproc_per_node=2 example/zero1/train.py

# ZeRO-2
torchrun --nproc_per_node=2 example/zero2/train.py

# ZeRO-3
torchrun --nproc_per_node=2 example/zero3/train.py
```

---

## 🧠 技术要点

* DeepSpeed ZeRO 并行机制（Stage 1/2/3）
* 分布式训练（DDP / 参数分片）
* Meta Device 模型初始化
* CPU-GPU Offloading
* 计算与通信重叠优化
* 大模型推理性能优化

---

## 📈 后续优化方向

* [ ] AMP 混合精度训练支持
* [ ] 多节点分布式训练（Multi-node）
* [ ] 通信 Bucket 优化
* [ ] KV Cache 优化（推理方向）
* [ ] MoE（Mixture of Experts）支持（规划中）

---

## 🙌 致谢

本项目基于原始 Tiny-DeepSpeed 实现进行增强开发，感谢原作者的开源贡献。
