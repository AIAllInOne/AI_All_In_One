AI All in One

## 基本概念

### 量化/Quantization

+ 减少内存占用
+ 加速计算
+ 减小功耗和延迟
Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32).
Reducing the number of bits means the resulting model requires less memory storage, consumes less energy (in theory), and operations like matrix multiplication can be performed much faster with integer arithmetic. It also allows to run models on embedded devices, which sometimes only support integer data types.

### 混合精度训练

<https://www.tensorflow.org/guide/mixed_precision?hl=zh-cn>
混合精度是指训练时在模型中同时使用 16 位和 32 位浮点类型，从而加快运行速度，减少内存使用的一种训练方法。通过让模型的某些部分保持使用 32 位类型以保持数值稳定性，可以缩短模型的单步用时，而在评估指标（如准确率）方面仍可以获得同等的训练效果。本指南介绍如何使用 Keras 混合精度 API 来加快模型速度。利用此 API 可以在现代 GPU 上将性能提高三倍以上，而在 TPU 上可以提高 60％。

### AMP

Automatic Mixed Precision
<https://pytorch.org/docs/stable/notes/amp_examples.html#>

## 硬件

### CPU

### GPU

### TPU

Tensor Processing Unit 张量处理单元 是一款为机器学习而定制的芯片，经过了专门深度机器学习方面的训练，它有更高效能（每瓦计算能力）。

## 架构

### ONNX/Open Neural Network Exchange

The open standard for machine learning interoperability

ONNX 是一种为表示机器学习模型而构建的开放格式。ONNX 定义了一组通用运算符（机器学习和深度学习模型的构建块）和一种通用文件格式，使 AI 开发人员能够将模型与各种框架、工具、运行时和编译器一起使用。

ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.
<https://onnx.ai/>

### ONNX runtime

<https://github.com/microsoft/onnxruntime>
ONNX Runtime: cross-platform, high performance ML inferencing and training accelerator

### TensorRT

<https://github.com/NVIDIA/TensorRT>
NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. This repository contains the open source components of TensorRT.

NVIDIA® TensorRT™ 是一款用于在 NVIDIA GPU 上进行高性能深度学习推理的 SDK。此存储库包含 TensorRT 的开源组件。

### NV Hopper

NVIDIA Hopper架构是NVIDIA在2022 年3月推出的GPU 架构。 这一全新架构以美国计算机领域的先驱科学家 Grace Hopper 的名字命名，将取代两年前推出的 NVIDIA Ampere 架构。 [1]

## 数据类型

### FP/BF

fp: float point 浮点精度
bf: brain float  
BF16是一种全新的数字格式，专门服务于人工智能和深度学习，最开始是Google Brain发明并应用在TPU上的，后来Intel，Arm和一众头部公司都在广泛使用。
以前大部分的AI训练都使用FP32，但有相关的论文证明，在深度学习中降低数字的精度可以减少tensor相乘所带来的运算功耗，且对结果的影响却并不是非常大。
FP32包含1（符号位）+ 8（指数位）+ 23（尾数位）
FP16包含1+5+10
BF16包含1+8+7

![FPBF](image/FPBF.webp)

BF是brain float的缩写，和缩小了动态范围的FP16相比，它不仅拥有和FP32相同的表示范围，且更少的尾数也使它能够拥有更小的芯片面积。

### E4M3/E5M2

FP8是FP16的衍生产物，它包含两种编码格式E4M3与E5M2。对于E4M3而言，其包含4比特指数、3比特底数、以及一比特符号位。E5M2同理包含5比特指数位、3比特底数、1比特符号。在本文中，我们称指数部分为exponent, 底数部分为mantissa。下图展示了FP32, FP16, FP8的格式对比：

![E4M3E5M2](image/E4M3E5M2.webp)

## 工具

### netron

深度学习可视化工具
<https://netron.app/>
![netron](image/netron.webp)

## 模型和参数

### Conv

convolution 卷积

### W

weight 权重

### B

Bias 偏差

### Gard

gradient 梯度

## 激活函数

### Relu

线性整流函数（Linear rectification function），又称修正线性单元，是一种人工神经网络中常用的激活函数（activation function），通常指代以斜坡函数及其变种为代表的非线性函数。
![relu](image/relu.webp)

### Sigmoid

![sigmoid](image/sigmoid.webp)

## 量化

### 对称量化和非对称量化

0点是否在原数据的零点
非对称量化: 偏移
![非对称量化](image/非对称量化.webp)

对称量化：最大绝对值对称法
![对称量化](image/对称量化.webp)
