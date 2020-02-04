# Visual Language Tasks Survey

## 前言

Visual Language任务指的是同时利用到视觉和文本信息且关注于建立起两者联系的任务，如VQA、Visual grounding等，本文侧重于记录此领域内的主要任务和一些值得关注的前沿方法。

### Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment 

#### Keywords

Visual grounding、Image Level监督

#### 解析

本文研究的是Visual grounding任务，使用Image level的监督数据，实现Image regions和Caption phrases之间的对齐。其提出早期方法直接用局部对齐的置信度来生产全局对齐的置信度，模型只需要学习判别一些关键区域的对齐就足以实现Image和Caption的对齐，这使得RoIs和短语的对齐训练不充分。作者考虑将模型认为对齐的部分抽取出来，仅使用对齐部分的feature作为输入，让Image Caption的对齐判断仅依赖于对齐的输入，这就要求网络需要先获取比较精细的对齐结果，这更加充分地利用了Image-Level的监督信号。

作者首先通过预训练网络获取Region of interests(RoIs)和Caption phrases和对应的特征向量，并进行使用过Image level监督实现这两者的对齐。

为此作者提出了三个模块：

* The Local Matching Module:对于给定的Caption请求，推断所有潜在的RoI-phrase对应关系
* The Local Aggregator Module:为上一模块的每一条对齐结果，生成caption-conditioned的image representation。
* The Global Matching Module:使用caption-conditioned的image representation判别与Caption的匹配程度。

##### The Local Matching Module

作者需要挖掘出潜在的RoI-phrase对应关系，所以对于每一个RoI $x_j$，其首先被投影到文本的特征空间，并用余弦相似度判断相关性。
$$
\begin{aligned}
\hat{x}_{j} &=W_{l}^{T} x_{j} \\
s_{j k} &=\frac{\hat{x}_{j}^{T} p_{k}}{\left\|\hat{x}_{j}\right\|_{2}\left\|p_{k}\right\|_{2}}
\end{aligned}
$$
传统方法会直接通过max取出phase对应的region，作者为了提高鲁棒性，考虑使用attention加权候选region，得到attended的region表示
$$
\begin{aligned}\alpha_{jk}&=softmax\left(s_{{1:R},k}\right)\\x^c_k&=\sum_j{\alpha_{jk}x_j}\end{aligned}
$$
用它作为captionc 中的phase k所对齐的RoI j的表示。当然如果仅仅需要获取对齐关系，那么只需要依据传统方法求max即可。

##### The Local Aggregator Module

对于图像I，其基于caption c的RoIs可以表示为$I^c_{rois}=\left(x^c_k\right)^P_{k=1}$，需要基于它获取caption-conditioned的image representation，作者直接用MLP+mean pooling的方式编码$I^c_{rois}$，写作$f_{enc}\left(I^c_{rois}\right)$。

##### The Global Matching Module

最终作者直接用RNN编码的caption representation和caption-conditioned的image representation进行余弦相似度比较得到相似度，训练时使用Rank Loss。





