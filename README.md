# qwen.metal

你可能试过[llama2.c](https://github.com/karpathy/llama2.c)，一个单文件C语言Llama推理实现，这里是一个支持Mac GPU加速的llama2.c，支持中文更强的Qwen模型本地推理。赶紧打开Mac试一下吧~~~

代码一共1500行（C和Metal），有以下特点：
* 没有外部依赖，只要有Mac就可以跑
* GPU加速具有较好的性能
* 基于llama2.c的代码，清晰易懂

## 小试牛刀

在Apple Silicon Mac上：

```
git clone https://github.com/llamajun/qwen.metal.git
cd qwen.metal
make
```

然后你需要下载模型权重，这里有一个我打好包的qwen1.5-0.5b-chat模型：[百度网盘](), [Google Drive]()。解压缩到qwen.metal目录下，然后运行：

```
make chat
```

就可以和模型进行对话了。

性能方面，在我的2021款MacBook Pro M1上面，可以跑到25 tokens/s。我们使用的是32位float计算，这个速度基本就把系统的内存带宽占满了（最大内存带宽是66.7 GB/s），所以是不错的性能。

## 推理Qwen1.5-7B-Chat

TODO

## LLM推理基础 🦙🦙🦙

Qwen的架构与LLAMA基本一致，有一些小的区别，比如self-attention计算的过程中，Qwen的计算是带有bias的，而Llama没有bias，只有weight。但总体上，类Llama模型的推理流程大致都是以下：

1. 首先将一堆权重矩阵读进来。
2. 自回归（autoregressive）计算的输入，是当前token的embedding。
3. 对于0.5B模型，一共有24个层，每层就是 normalize -> self-attention (包含RoPE) -> feed-forward network(ffn) 三步。
4. 所有层做完后，最后再进行 normalize -> final ffn 两步，就得到本次自回归的logits。
5. 通过采样（top_p sampling）步骤，就生成下一个token，再回到第2步。

推理过程确实挺简单，比之前的encoder-decoder模型更简化了。一个Llama模型架构的不错的介绍是[这篇](https://zhuanlan.zhihu.com/p/651248009)。

## 设计说明

欢迎复用这个代码，所以写一点设计说明。关于[llama2.c](https://github.com/karpathy/llama2.c)的情况和设计，可以看作者Andrei Karpathy的原代码repo。这里说一些增加的内容。

### 主要的优化都针对GEMV

GEMV（矩阵向量乘法）是Llama/Qwen推理中最主要的运算，就是llama2.c中的`matmul()`。如果我们在服务器环境下，同时服务多个客户端的话，那通过合并多个请求，也就是batching，可以将GEMV转化成效率更高的矩阵矩阵乘法GEMM。但在端侧推理中，因为只有一个客户端，batching基本上帮不上忙，计算被GEMV主导。因此我们从GEMV开始做一些性能优化。

GEMV可以进行的优化是比较单纯的，大部分情况下GEMV都受限于内存带宽（memory-bound），因此我们的目的很简单，就是尽量用满内存带宽，而且不要浪费，就基本能达到GEMV的较好性能了。具体代码中在`llm-metal.metal`里的`gemv()`，做了两个简单的优化：

1. 用`float4`类型，一次计算4个float数，这样每个GPU线程的效率更高。
2. 按行和列都做切分，保证有足够多的线程，能将访存的延迟隐藏起来，用满带宽。每行这里分成32个线程，正好是GPU的执行宽度，保证这32个线程一定在同一个SIMD group里面，这样整个行的求和可以直接用`simd_sum()`一步完成。

Metal的性能分析用Instruments可以很方便进行。下面是最初版本的内存利用率（一个token的计算），可以看到只有最后一小段用满带宽：

![](doc/metal-bandwidth.png)

GEMV优化之后（目前版本）的内存利用率，可以看到大部分时候都能用满带宽了：

![](doc/metal-bandwidth2.png)

### 让绝大部分计算在GPU上发生

在做手机端和IoT上的AI推理的时候，有一个经常发生的问题，就是CPU和GPU之间交替计算的时候，性能会很差，因为同步的时间和数据的传输都带来开销。在这里，因为桌面GPU通用性比较强，使得这里我们比较容易（仅仅写了8个GPU shader函数，见`mtl-metal.metal`）就基本实现了全GPU的计算。

看`run.c`中的`forward()`，这是Qwen推理的最主要代码，在这里调用了`lm_gemv()`, `lm_rmsnorm()`, `lm_multihead_attention()`等一批计算函数，这些计算都在GPU上面发生。Metal API中有一个基本部件是command queue，GPU可以同时执行多个命令（command），而command queue就是用来管理这些Command。在`llm-metal.m`中，当`lm_gemv()`这些调用发生时，我们仅仅是将这些命令加入到命令队列中，然后当调用`lm_execute()`的时候，一次性执行这些命令。

`lm_execute()`每产生一个token调用一次，所以所有的几十个层的计算的工作，是一次性提交到GPU上，然后全部计算完成后，再回到CPU来进行后面的工作的。这样CPU和GPU的交互粒度比较大，大幅减少了CPU和GPU调度和交换数据的消耗。

`llm-metal.h`是主程序`run.c`和Metal部分之间的接口，我们通过buffer和command queue的管理，分开了Metal和非Metal代码，这样更简洁。后面如果要支持非Apple GPU的硬件，主程序也不用怎么修改。

### 使用Apple GPU和Metal的评价

Metal API使用起来非常简单，Apple的优势就是软硬件高度集成，不需要安装驱动，直接就可以跑。再加大带宽的unified memory，目前Mac变成了端侧大模型的重要平台，llama.cpp，llamafile等重要开源项目，都在Mac上非常活跃。

Apple的GPU是他们自己的IP，总体上的文档是比较不足的，通过找网上的一些资料，有一些简单的发现：
  * Warp大小是32，也就是一个GPU单元，同时可以执行32个线程，如果卡住了（比如读内存），就调度另外32个线程过来继续执行。
  * 能调度的线程来自同一个threadgroup，一个threadsgroup最大1024个线程。
  * 规模最小的M1 GPU同时跑24个threadgroups，其它的GPU则可以跑更多。按这个计算，如果要填满整个GPU，需要24*1024=24576个线程。
  * Apple GPU 16位性能比较强，但这里没有用到，全是32位计算。
  * 对于本地推理，如前面所说内存带宽是瓶颈，正确方法应该是4位或者8位存储，然后16位或者32位计算（因为计算利用率很低，所以对性能应该不会有影响）。




