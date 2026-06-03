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
```
## 4. 第 12 周更新说明

本周在第 11 周 baseline 的基础上，继续完善了基于对抗训练自编码器的非对称图像加密原型。主要更新内容如下：

### 4.1 模型结构更新

- 加深了 PrivGen 私钥生成网络，由简单映射扩展为多层全连接网络，并加入 Dropout，提高密钥表示能力。
- 加深了 PubGen 公钥生成网络，使公钥由随机向量和私钥共同生成，增强公钥与私钥之间的关联表达。
- 对 Alice、Bob、Eve 三个模块进行结构扩展：
  - Alice 在多层中反复引入公钥信息，用于生成密文向量；
  - Bob 在多层中反复引入私钥信息，用于从密文中恢复原图；
  - Eve 在多层中反复引入公钥信息，用于模拟攻击者在无私钥条件下的重构能力。

### 4.2 训练流程更新

- 保留 Alice、Bob、Eve、KeyGen 的对抗训练框架。
- 每个 batch 中先固定 Alice/KeyGen/Bob，多步训练 Eve，使 Eve 尽可能从密文和公钥中恢复原图。
- 再固定 Eve，训练 Alice/KeyGen/Bob，使 Bob 的重构误差降低，同时尽量增大 Eve 的攻击误差。
- 当前 Alice/KeyGen/Bob 的优化目标为：

```text
loss_AKB = L_b - mu * L_e
```

## 第 13、14 周补充实验：嵌入向量可视化对比

为了分析普通自编码器和对抗加密自编码器在编码空间中的差异，本周增加了 embedding 可视化实验。

实验方法如下：

1. 训练一个普通 AutoEncoder，提取其 Encoder 输出的 latent vector；
2. 使用当前对抗加密模型，提取 Alice 输出的 ciphertext vector；
3. 分别使用 PCA 和 t-SNE 将两类向量降维到二维空间；
4. 使用 MNIST 数字标签进行着色，观察不同数字类别在编码空间中的聚类情况；
5. 计算 silhouette score 和 kNN label accuracy，辅助衡量 embedding 中保留的类别信息。

实验预期：

- 普通 AutoEncoder 的 latent vector 通常保留较多图像语义，因此不同数字类别可能形成较明显的聚类；
- 对抗加密模型中 Alice 输出的 ciphertext 需要隐藏原图信息，使 Eve 难以攻击恢复，因此其二维可视化结果中不同类别应更加混合；
- 若 adversarial ciphertext 的 silhouette score 和 kNN label accuracy 明显低于普通 AutoEncoder，则说明该编码空间中可直接恢复的类别信息更少，具有一定的信息隐藏效果。
- 
<center class ='img'>
<img title="普通编码器潜空间" src="plain_ae_tsne" width="45%">
<img title="加密编码器潜空间" src="adversarial_cipher_tsne" width="45%">
</center>

