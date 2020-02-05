# Visual Language Tasks Survey

## 前言

Visual Language任务指的是同时利用到视觉和文本信息且关注于建立起两者联系的任务，如VQA、Visual grounding等，本文侧重于记录此领域内的主要任务和一些值得关注的前沿方法。

## 任务

### Visual Textual Alignment

对图像和文本模态的数据进行对齐，其根据被对其的层次还可以被进一步细分。

#### Image Text Matching

对完整图片和Caption的匹配，还会衍生出Image2Caption和Caption2Image。

#### Region Phrase Matching

图像区域和文本短语的匹配。

## 方法

### Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering

#### Keywords

Attention、跨模态表示学习

#### 解析

作者基于自然语言处理中的Transformer模型，提出了一种跨模态的Co-Attention模型，以图片和其描述为例，抽取出Regions和Phrases序列，可以分别生成每个Phrase对所有Regions的attended feature和Region对所有Phrases的attended feature。

为了表述方便，这里用$Q_l$指代words feature序列并用$V_l$指代pixel feature(经过卷积抽取过的)序列，作者堆叠了Co-attention并用$l$表示层数。

##### Co-attention

$Q_l \in \mathbb{R}^{d\times N}\quad V_l \in \mathbb{R}^{d\times T}$，其中$N,T$分别表示单词数和像素点数，$d$表示channel数。由于其中存在拼接、相加的操作，所以必须使得两者具有同样的channel数。

![image-20200205220237786](imgs\image-20200205220237786.png)

作者首先计算Affinity matrix
$$
A_l=V_l^{\mathrm T} W_l Q_l
$$
其中$W_l$将两者投影到同一潜空间。$A_l \in \mathbb{R}^{T\times N}$，保存了pixels和words之间的吸引性，作者随后分别在row-wise\column-wise进行softmax。
$$
\begin{aligned}
A_{Q_l}&=softmax\left(A_l\right)\\
A_{V_l}&=softmax\left(A_l^{\mathrm T}\right)
\end{aligned}
$$
$A_{Q_l}$表示的是每个pixel对各个word的attention map，用英文标识是attention maps on words conditioned by pixel, namely $P\left(word|pixel\right)$。$A_{V_l}$则是word对pixel的attention map。

作者进而生成了attended feature $\hat{Q_l}\,and\,\hat{V_l}$
$$
\begin{aligned}
\hat{Q_l}&=Q_l A_{Q_l}^{\mathrm T}\\
\hat{V_l}&=V_l A_{V_l}^{\mathrm T}
\end{aligned}
$$
本文有趣的地方来了，$\hat{Q_l}\in \mathbb{R}^{d\times T}$，其是由文本模态特征生成的每个pixel的特征表示，换一种理解，由于获取了每个pixel对各个word的attention，那么就可以通过fuse all word representation来得到pixel的文本表示，其是文本特征与视觉结构（未用到任何视觉特征）的结合，$\hat{V_l}\in \mathbb{R}^{d\times N}$同理。

作者考虑到并不是所有pixel或word都能找到对应的，这种nowhere element会影响到模型性能，为此作者给$V_l,Q_l$分别增加了$K$个空元素得到$\tilde{Q_l} \in \mathbb{R}^{d\times \left(N+K\right)},\tilde{V_l} \in \mathbb{R}^{d\times \left(T+K\right)}$。
$$
\begin{aligned}
\hat{Q_l}&=\tilde{Q_l} A_{Q_l}\left[1:T,:\right]^{\mathrm T}\\
\hat{V_l}&=\tilde{V_l} A_{V_l}\left[1:N,:\right]^{\mathrm T}
\end{aligned}
$$
最终得到的两个表示维度不变。

![image-20200205215818181](imgs\image-20200205215818181.png)

作者进而对原特征和attended特征进行融合，如上图所示
$$
\begin{aligned}
Q_{l+1}&=\mathrm{ReLU}\left(W_{Q_l}\left[Q_l,\hat{V_l}\right]+b_{Q_l}\right)+Q_l\\
V_{l+1}&=\mathrm{ReLU}\left(W_{V_l}\left[V_l,\hat{Q_l}\right]+b_{V_l}\right)+V_l
\end{aligned}
$$
其中$W_{Q_l},W_{V_l} \in \mathbb{R}^{d\times 2d}$用于fuse拼接的两者。

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

![image-20200205214846947](imgs\image-20200205214846947.png)

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





