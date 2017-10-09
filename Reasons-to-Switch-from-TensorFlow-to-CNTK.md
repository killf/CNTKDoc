---
title:   为什么要从TensorFlow迁移到CNTK
author:    killf
date:	2017-10-09
---
# 为什么要从TensorFlow迁移到CNTK

深度学习在过去的几年里彻底改变了人工智能（AI）领域。微软的愿景是希望所有人都能够使用智能，而不仅仅是几家精英公司，为此开发了Microsoft Cognitive Toolkit（CNTK，微软认知工具箱）。这是一款免费的、可供任何人使用的、开源的深度学习工具包，它现在是GitHub上第三大热门的机器学习工具包，仅位于TensorFlow和Caffe之后（在MxNet、Theano、Torch等之前）。

在TensorFlow流行的当下，我们经常会被问及：为什么有人想使用CNTK而不是TensorFlow？人们都有趋众心理，这当然没有问题。不过，本文想指出一些有利于CNTk的强有力的证据，并且在许多应用中，CNTK可能是一个更好的选择。这些原因包括：

- **速度**。 CNTK通常比TensorFlow快很多，特别是在循环网络上，可以快5-10倍。
- **准确性**。 CNTK可用于训练具有最高精度的深度学习模型。
- **API设计**。 CNTK具有非常强大的C++ API，并且还具有低级的和易于使用的高级API（Python）。
- **可伸缩性**。 CNTK可以轻松地扩展到拥有数千个GPU的平台上。
- **评估**。 CNTK拥有C＃/.NET/Java等多平台的评估接口，可以方便地将CNTK的评估功能集成到用户应用程序中。
- **可扩展性**。 CNTK可以方便地扩展layer和learner。
- **内建的数据读取器**。 CNTK内置了多种高效的数据读取器（支持分布式学习）。

本文的其他部分会详细的介绍和解释这些优点。

## 原因1：速度

深度学习是数据密集型和计算密集型的任务，无论是构建产品还是撰写论文，速度都是关键。在学习和评估的速度方面，CNTK具有优于TensorFlow的优势。香港浸会大学（http://dlbench.comp.hkbu.edu.hk/）研究人员的论文（https://arxiv.org/pdf/1608.07249.pdf）显示，在他们测试的所有网络中，CNTK的性能都比TensorFlow好，无论是CPU还是GPU。事实上，在GPU上，CNTK比其他所有的工具都要快。

对于图像相关的任务，CNTK通常比TensorFlow快2到3倍。而对于循环神经网络（RNN），CNTK是无可争议的赢家。如上述文章所述，当在CPU上运行时，“CNTK比Torch和TensorFlow获得更好的性能（高达5-10倍）”。而在GPU上，“CNTK多次超越所有其他工具”。这个速度优势不是偶然的，微软研究院在CNTK的序列处理方面做了大量的优化。

如果您的项目涉及序列处理，如语音、自然语言理解、机器翻译等，CNTK是您的最佳选择。如果您是视频处理的视觉研究人员，那么您一定得尝试一下CNTK。

## 原因2：准确性

深度学习工具包很难开发，因为即使工具包中存在Bug，您也可以通过设计网络架构来实现合理的准确性。许多其他工具包的示例代码与论文的原始实现几乎相同，然后提供一个与训练的模型供人们下载和使用，这在我们看来是非常不负责任的。在CNTK中，我们非常注意错误跟踪，并确保该工具包可以用于从头开始训练模型，并获得最高的准确性。

最初由Google研究人员开发的Inception V3网络（https://arxiv.org/abs/1512.00567）便是一个例子。 TensorFlow共享了Inception V3的训练脚本，并提供训练好的模型进行下载。然而，难以重新训练模型并达到相同的准确性，因为这需要对诸如数据预处理和增强之类的细节的额外理解。原论文的第三方报告（在Keras上）所达到的最佳准确度大约为0.6％。 经过CNTK团队的努力研究，CNTK已经能够训练一个top5错误为5.972％的Inception V3模型，甚至比原论文更好！训练脚本以示例的方式（https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Classification/GoogLeNet）共享，您可以自己验证。

对于循环神经网络，CNTK的自动批处理算法可以包装不同长度的序列并拥有很高的执行效率。更重要的是，它能够更好地随机初始化训练数据，与对原始数据的封包相比，通常可以提高1-2％的准确性。这使得微软研究院的语音小组在语音识别中获得更人性化的结果（https://arxiv.org/pdf/1610.05256.pdf）。

## Reason 3: API design
TensorFlow was originally designed to have a small C++ core API, and most of its functionality is implemented in Python. This certainly has advantages, since Python is much easier to write and faster to update. The downside, however, is multiple. We already mentioned its lower speed. In addition, for many real-world applications, it is not always possible to embed Python code if the application itself is written in C++ and has tough restrictions in running time. In TensorFlow 1.0, an extensive C++ API has been released, although it remains to be somewhat slow in performance.

We envision from the very beginning that not only the evaluation of deep learning models would be an essential part of applications. The model training part can also be closely integrated into intelligent applications such as Office or Windows. CNTK is designed to have almost all functionalities written in C++. This not only makes it extremely fast, but also allows it to be used as a C++ API ready to be integrated with any applications. It also makes it very easy to add additional bindings to CNTK, such as Python, R, Java, etc.

It is also important to mention that CNTK’s Python API contains both low-level and high-level implementations. The high-level Python API is designed with a functional programming paradigm, and is highly compact and intuitive, especially when dealing with recurrent neural networks. This is in contrast with TensorFlow’s Python API, which is considered too low level by most people. TensorFlow users rely on third party high level APIs such as TensorFlow Slim and Sonnet to fill the gap, but the space is now fragmented due to too many choices.

## Reason 4: Scalability
Modern deep learning tasks could easily face billions of training examples. Being able to scale across multiple GPUs and multiple machines is a necessity. Toolkits such as TensorFlow can handle multiple GPUs on a single machine. However, when there is need to scale across multiple machines, it often becomes too complicated to fulfill.

In contrast, distributed training has always been a strong selling point for CNTK. Changing a training script from single GPU to multiple GPUs across multiple machines is merely a few lines of change, as seen in many examples in the CNTK repository. Inside Microsoft, CNTK has been used to train tasks across hundreds of GPUs on many machines. We have invented some of the most innovative schemes for parallel training, such as 1-bit SGD (http://research.microsoft.com/apps/pubs/?id=230137) and Block-Momentum SGD (https://www.microsoft.com/en-us/research/publication/scalable-training-deep-learning-machines-incremental-block-training-intra-block-parallel-optimization-blockwise-model-update-filtering/). These algorithms helped improve hyper-parameter tuning significantly, leading to better models in shorter time, such as the conversational speech recognition breakthrough by Microsoft Research (https://arxiv.org/pdf/1610.05256.pdf).

## Reason 5: Inference
TensorFlow has a very nice story about serving. It supports multiple versions of models, saving model optimized for serving, multiple meta graphs inside the same model to support serving on different devices, and plug-ins to support user customization. With their XLA AoT compilation, TF can transform the model into executable, which significantly reduces model sizes for mobile and embedded devices and latency.

Compared to TensorFlow, CNTK is more focusing on direct integration of CNTK Eval into user applications. Besides Python and C++, CNTK provides C#/.NET API for inference. The C#/.NET API is built directly upon the C++ API with least performance overhead. Java API will be available soon. If you are building a .NET application and would like to choose a deep learning toolkit for inference, CNTK is a much more natural choice than TensorFlow.

CNTK also supports parallel evaluation of multiple requests with very limited memory overhead. This provides great advantage for deploying models in a service environment like Web application. For edge devices, CNTK supports both Intel and ARM platform.

## Reason 6: Extensibility
TensorFlow is a very flexible toolkit, and one can almost implement any model with it. However, if you are a current Caffe user, you are out of luck. There is no easy way to convert your Caffe script to TensorFlow, other than rewriting everything from scratch. Similarly, if you would like to try a new layer invented by someone else and written in a different toolkit, your options are either to implement it yourself, or to wait for someone else to implement it for you.

CNTK is arguably the most extensible toolkit available as there exists no such restriction. Through UserFunctions, any operator can be implemented in pure Python. Having NumPy arrays as the interface between core CNTK and the Python bindings, you simply implement the forward and backward pass, and the newly created operator can be immediately placed into the graph. Furthermore, it is straightforward to even place other toolkit’s graph execution into a CNTK UserFunction and thus speed up the experimentation phase tremendously.

Same applies to the gradient update algorithms. Although CNTK provides most algorithms like RMSProp or Adam out of the box, you have full freedom to implement new learning approaches in pure Python.

## Reason 7: Built-in readers
Deep learning shines when there's a lot of training data. For some applications, the data is so large that it does not fit in RAM and sometimes not even on a single machine. Even when the data fits in RAM, a naïve training loop will spend a lot of its time transferring data from RAM to GPU. CNTK's built-in readers solve all the above problems by providing highly performant facilities for iterating through a dataset, without the need for the data to fit in RAM. They can be used either with a single disk or a distributed file system such as HDFS. Prefetching is used extensively so that the GPU is always utilized without stalling. CNTK's readers can also ensure that the model always receives the data in a properly shuffled order (that improves convergence) even if the underlying dataset is not shuffled. Finally, all these facilities are available to all current and future readers. Even if you write a reader for your exotic file format, you won't have to worry about implementing prefetching yourself.

To conclude the article, we promise you that if you adopt CNTK, you will be using exactly the same toolkit used by Microsoft product groups, such as Bing, Cortana, Windows, etc. We hope this will give you confidence that you are not compromised in any way because you choose CNTK. We welcome your contribution to CNTK, and together we make it the deep learning toolkit that truly democratize AI.
