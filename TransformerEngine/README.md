# TransformerEngine

https://github.com/NVIDIA/TransformerEngine

Transformer Engine (TE) 是一个用于在 NVIDIA GPU 上加速 Transformer 模型的库，包括在 Hopper GPU 上使用 8 位浮点 (FP8) 精度，以在训练和推理中以更低的内存利用率提供更好的性能。TE 为流行的 Transformer 架构提供了一组高度优化的构建块，以及一个可与特定于框架的代码无缝使用的自动混合精度 API。TE 还包括一个与框架无关的 C++ API，可以与其他深度学习库集成以启用对 Transformer 的 FP8 支持。

随着 Transformer 模型中参数数量的不断增长，BERT、GPT 和 T5 等架构的训练和推理变得非常耗费内存和计算资源。大多数深度学习框架默认使用 FP32 进行训练。然而，这对于实现许多深度学习模型的完全准确性来说并不是必不可少的。使用混合精度训练，即在训练模型时将单精度 (FP32) 与较低精度 (例如 FP16) 格式相结合，与 FP32 训练相比，可显著提高速度，同时准确度差异很小。Hopper GPU 架构引入了 FP8 精度，与 FP16 相比，其性能有所提高，而准确度没有降低。虽然所有主要的深度学习框架都支持 FP16，但目前的框架并不原生支持 FP8。

TE 通过提供与流行的大型语言模型 (LLM) 库集成的 API 来解决 FP8 支持问题。它提供了一个 Python API，其中包含可轻松构建 Transformer 层的模块以及 C++ 中与框架无关的库，包括 FP8 支持所需的结构和内核。TE 提供的模块在内部维护 FP8 训练所需的缩放因子和其他值，大大简化了用户的混合精度训练。