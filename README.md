# 基于对抗训练自编码器的非对称图像加密原型

## 1. 项目简介

本项目是《Python人工智能程序设计实践》课程项目，题目为“基于对抗训练自编码器的非对称图像加密原型”。

项目尝试使用深度学习方法构建一个简化的神经网络图像加密与解密系统。整体结构包含 Alice、Bob、Eve 和密钥生成模块：Alice 使用公钥将图像加密为密文，Bob 使用私钥从密文中恢复原图，Eve 在没有私钥的情况下尝试攻击并重构原图。

当前版本主要目标是完成一个可以运行的 baseline，验证“密钥生成—加密—解密—攻击—反向传播”这一完整训练闭环。

## 2. 当前阶段说明

当前为第 11 周代码仓库更新版本。根据第 10 周中期进展，本项目已经完成：

- MNIST 数据集读取与预处理；
- 图像张量化与归一化；
- PrivGen / KeyGen、Alice、Bob、Eve 等基础网络模块；
- 对抗训练流程；
- Bob 重构误差与 Eve 攻击误差记录；
- TensorBoard 训练日志输出；
- 原图、Bob 重构图、Eve 重构图的可视化保存；
- 模型权重保存。

当前版本仍属于原型阶段，重点是跑通训练流程，而不是证明模型已经具备严格密码学安全性。

## 3. 项目结构

```text
asymmetric-image-encryption/
├── README.md
├── asymmetric_encrypter.py
├── requirements.txt
├── .gitignore
├── sample_results.png
└── logs/