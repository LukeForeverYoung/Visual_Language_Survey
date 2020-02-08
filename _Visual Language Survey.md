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

### Auto-Encoding Scene Graphs for Image Captioning

2019 CVPR

#### Keywords

GCN、跨模态、自编码器

#### 解析

关于Caption生成任务，一直以来存在一个问题就是语言缺少归纳性，把目标检测的结果拼凑得到文本语义往往比较简单，而人类在描述时会基于一些预先学习的假设(inductive bias)，得到高度概括的语句。因此作者提出一种图卷积+自编码器的方法挖掘更深层次的语义并进行归纳总结，以提升Caption的效果。

##### Encoder-Decoder

 作者的Image2Caption编码过程表示如下
$$
\begin{aligned}
\mathrm{Encoder:}&\,\mathcal{V}\leftarrow\mathcal{I}\\
\mathrm{Map:}&\,\hat{\mathcal{V}}\leftarrow\mathcal{R\left(V,G;D\right)},\,\mathcal{G}\leftarrow\mathcal{V}\\
\mathrm{Decoder:}&\,\mathcal{S}\leftarrow\hat{\mathcal{V}}
\end{aligned}
$$
作者首先抽取图像特征$\mathcal{V}$并构建Graph，再将两者和字典$\mathcal{D}$一同用于re-encoder过程，得到$\mathcal{\hat{V}}$，进而解码得到句子。

对于字典$\mathcal{D}$，其是一个动态的记忆字典，挖掘了自然语言中的inductive bias，作者使用一个Sentence2Sentence的无监督自学习过程进行训练。
$$
\begin{aligned}
\mathrm{Encoder:}&\,\mathcal{X}\leftarrow\mathcal{G}\leftarrow\mathcal{S}\\
\mathrm{Map:}&\,\hat{\mathcal{X}}\leftarrow\mathcal{R\left(X;D\right)}\\
\mathrm{Decoder:}&\,\mathcal{S}\leftarrow\hat{\mathcal{X}}
\end{aligned}
$$
![image-20200207165456718](imgs\image-20200207165456718.png)

##### Auto-Encoding Scene Graphs

首先作者从句子中抽取出若干objects, attributes and relationships记为$o_i,a_{i,l},r_{ij}$，并将它们作为Graph中的node，并根据以下规则建立有向边。

* $a_{i,l}\rightarrow o_i$
* $o_i\rightarrow r_{ij}\rightarrow o_j$

在GCN中，作者分别设计了四个空间卷积函数分别对这三种节点进行编码：

* Relationship编码：$x_{r_{ij}}=g_r\left(e_{o_i},e_{r_{ij}},e_{o_j}\right)$
* Attribute编码：$x_{a_i}=\frac{1}{N_{a_i}}\sum^{N_{a_i}}_{l=1}{g_a\left(e_{o_i},e_{a_{i,l}}\right)}$
* Object编码：$x_{o_i}=\frac{1}{N_{r_i}}\left[\sum_{o_j \in sbj\left(o_i\right)}{g_s\left(e_{o_i},e_{r_{ij}},e_{o_j}\right)}\\+\sum_{o_k \in obj\left(o_i\right)}{g_o\left(e_{o_i},e_{r_{ik}},e_{o_k}\right)} \right]$

![image-20200207171313430](imgs\image-20200207171313430.png)

前两者易理解，关于Object编码，作者认为object与其他object的上下位关系(subject/object)需要单独考虑，因此设计了两个卷积函数。

对于$\hat{\mathcal{X}}\leftarrow\mathcal{R\left(X;D\right)}$部分，作者将$\mathcal{D}$表示为矩阵$D\in \mathbb{R}^{d\times K}$，其中$K$表示记忆的容量。作者通过Attention的方式生成基于Memory的attended表示
$$
\begin{aligned}
\alpha&=\mathrm{softmax}\left(D^{\mathrm{T}}x\right)\\
\hat{x}&=R\left(x;D\right)=D\alpha=\sum^K_{k=1}{\alpha_k d_k}
\end{aligned}
$$
事实上，新生成的$\hat{x}$完全依赖于$D$的内容以及$x$对于$D$的Attention，这就会迫使$\mathcal{D}$学习到高度概括性的语义。

Auto-Encoding Scene Graphs(SGAE)的训练过程是无监督的，并且可以无限迭代。

##### Multi-modal Graph Convolution Network

Image2Caption部分的视觉特征$\mathcal{V}$同样可以生成Graph，作者将文本模态(label embdding $e_{o_i}$)和视觉模特(visual embedding $v_{o_i}$)生成Graph的节点编码进行融合

$u_{o_i}=\mathrm{ReLU}\left(W_1 e_{o_i}+W_2 v_{o_i}\right)-\left(W_1 e_{o_i}+W_2 v_{o_i}\right)^2$

再使用与SGAE相同的方式生成Sentence。

### Parallel Attention: A Unified Framework for Visual Object Discovery through Dialogs and Queries

2018 CVPR

#### Keywords

Attention、Visual Grounding、Object Recognize

#### 解析

本文针对的任务是Visual Grounding任务，给出一张图片和一些有关文本，需要找出其匹配的图像区域。

作者提出了一个基于对话和询问的目标认知模型，提出了一种不需要固定标签集合的目标认知方法。其认为模型在识别目标时，首先应该关注到大体的候选区域，在读取自然语言输入时一步步缩小范围，最终收敛到一个局部区域。

为此作者提出了一个two-way attention mechanism

* Image-level attention: 建立文本上下文信息与图片的联系（如场景）
* Propose-level attention: 建立候选区域与自然语言中关键词的联系。

![image-20200207204239923](imgs\image-20200207204239923.png)

注：自然语言输入有两种，一种是问答序列，另一种是一个句子，在图片中，$d_1,...d_L$并不是说模型需要同时接收这两者，而是当输入为问答序列时，每次读取一个问答对并走橙色路线，反之每次读取一个单词，走蓝色路线。

##### Image-level attention

原图像首先通过CNN抽取得到视觉特征$V \in \mathbb{R}^{K,d}$，$K$表示抽取后的像素数目。在每个时间步，视觉特征和LSTM的隐藏状态通过计算attention进行融合求得attended visual feature $z_t$。随后输入的文本特征$m_t$和视觉特征$z_t$会更新隐藏状态$h_{t-1}$得到$h_t$。
$$
\begin{aligned}
e_{ti}&=\mathrm{tanh}\left(W_v v_i+W_h h_{t-1}\right)\\
\alpha_{ti}&=\mathrm{softmax\left(e_{ti}\right)}\\
z_t&=\sum^K_{i=1}{\alpha_{ti}e_{ti}}\\
h_t&=\mathrm{LSTM}\left(m_t,z_t,h_{t-1}\right)\\
\end{aligned}
$$
随着LSTM不断读取文字，attended视觉表示$z_t$会更加关注于文本涉及的内容，最后得到的$h_{t=L}$会作为image-level attention的输出。

##### Propose-level attention

对于候选区域的特征生成，作者考虑了三个部分：视觉特征$u$、空间特征$s$、标签特征$c$，每个区域的特征标记为$p_i=\left[u_i,s_i,c_i\right]$。

与image-level相同，作者使用完全相同的attention方法进行编码。
$$
\begin{aligned}
e^{'}_{ti}&=\mathrm{tanh}\left(W_p p_i+W^{'}_h h^{'}_{t-1}\right)\\
\beta_{ti}&=\mathrm{softmax\left(e^{'}_{ti}\right)}\\
z^{'}_t&=\sum^K_{i=1}{\beta_{ti}e^{'}_{ti}}\\
h^{'}_t&=\mathrm{LSTM}\left(m_t,z^{'}_t,h^{'}_{t-1}\right)\\
\end{aligned}
$$
但在最后，作者需要的是attended的propose 序列，所以其输出为
$$
\tilde{p}_i=\beta_{Li}p_i
$$

##### Referring

作者通过让每个候选region的最终表示$\tilde{p}_i$和image-level attention的表示进行点积，并通过softmax得到区域选中的概率分布。使用交叉熵损失训练。
$$
P=\mathrm{softmax}\left(h_{t=L}\odot\tilde{P}\right)
$$
这一方法并不需要人为标注目标是什么，但需要给出候选区域（作者使用交叉熵而非IoU）标注，而且多轮对话实际运用中也没那么理想。

### Visual Semantic Reasoning for Image-Text Matching

2019 ICCV

#### Keywords

Text-Image Matching、视觉推理、弱监督

#### 解析

本文应用的任务也是图像和文本的匹配任务，作者考虑到视觉图像中除了Object和Sense外，它们之间的interaction, relative positions等high-level的语义没有没很好地考虑到，所以作者提出了一个视觉表示学习模型，其能够捕获到Objects和它们的语义关系。

本文的融合了GCN做局部推理以及LSTM做局部特征的融合（也被称作全局推理）这两种思路。

![image-20200208174610546](imgs\image-20200208174610546.png)

##### Image Representation by Bottom-Up Attention

作者首先通过Bottom-Up Attention方法从图像$I$中获取region representations $V=\left\{v_1,...,v_k\right\}$。

Bottom-Up Attention是2018 CVPR的Oral，本文也会对其进行讲解。

##### Region Relationship Reasoning

作者为每一张图片的regions建了一张完全图，其通过计算regions之间的affinity matrix得到region之间的关系
$$
\begin{aligned}
\varphi\left(v_i\right)&=W_\varphi v_i\\
\phi\left(v_j\right)&=W_\phi v_j\\
R\left(v_i,v_j\right)&=\varphi\left(v_i\right)^{\mathrm{T}}\phi\left(v_j\right)\\
\end{aligned}
$$
随后使用GCN实现相关regions之间的信息传递
$$
\begin{aligned}
V^*=W_r\left(RVW_g\right)+V
\end{aligned}
$$
$W_g$是GCN权重，$W_r$是残差权重。作者这样做的好处是节点之间的连接关系没有固定，是可学习的。得到的$V^*=\left\{v^*_1,...,v^*_k\right\}$即relationship-enhanced表示。

##### Global Semantic Reasoning

作者通过一个RNN对特征序列进行fuse。由于每个region的特征都被其所关联的region增强，所以每个时间步输入进RNN的本质上是当前region及其相关region的融合。作者通过上一步的记忆和当前的图像特征生成update memory
$$
\begin{aligned}
r_i&=\sigma_r\left(W_r v^*_i+U_r m_{i-1}+b_r\right)\\
\tilde{m}_i&=\sigma_m\left(W_mv^*_i+U_z\left(r_i\circ m_{i-1}+b_m\right)\right)
\end{aligned}
$$
其中$\circ$表示element-wise multiplication，$r_i$表示reset gate以丢弃掉不需要的记忆。随后此时间步的记忆由上一步的记忆和更新记忆带权相加得到。
$$
\begin{aligned}
z_i&=\sigma_z\left(W_z v^*_i+U_z m_{i-1}+b_z\right)\\
m_i&=\left(1-z_i\right)\circ m_{i-1}+z_i\circ \tilde{m}_i
\end{aligned}
$$
最终的$m_K$作为图像$I$最终的视觉表示。

#####  Learning Alignments by Joint Matching and Generation

最后作者提出了两个Loss任务共同训练。

* Image-Text Matching任务：使用了hinge-based tripet ranking loss with emphasis on hard negatives。它每次只对最困难的负样本予以相应。$+$表示取max。

$$
L_M=\left[\alpha-S\left(I,C\right)+S\left(I,\hat{C}\right)\right]_++\left[\alpha-S\left(I,C\right)+S\left(\hat{I},C\right)\right]_+
$$

* Caption Generation任务：使用log似然损失

$$
L_G=-\sum^l_{t=1}{\log{p\left(y_t|y_{t-1},V^*;\theta\right)}}
$$



### Multi-level Multimodal Common Semantic Space for Image-Phrase Grounding

2019 CVPR

#### Keywords

Layer Attention、跨模态

#### 解析

这篇文章提出了一个新视角，将文本对视觉的注意力应用在视觉特征抽取的过程中，而非仅对抽取结果进行Attention。

##### Multi-Level Multimodal Attention Mechanism

![image-20200206171423397](imgs\image-20200206171423397.png)

对于视觉特征抽取，作者将预训练网络中的$L$层中间特征（包括最后一层）抽出并上采样到相同分辨率，再用$1\times 1$卷积得到同样大小的特征向量，组合后得到$V \in \mathbb{R}^{N\times L\times D}$，其中$N=M\times M$是中间特征统一后的分辨率。

文本特征表示为$S\in \mathbb{R}^{T\times D}$，生成过程略过，每个词的表示都被正则成单位向量。

作者首先计算两者的Heat map，并通过Heat map融合掉$V$中的channel$N$，以得到word对每一层的attened表示并转化为单位向量。
$$
\begin{aligned}
H_{n,l,t}&=max\left(0,\left<s_t,v_{n,l}\right>\right)\\
a_{t,l}&=\frac{\sum^N_{n=1}{H_{n,l,t}v_{n,l}}}{\left\|\sum^N_{n=1}{H_{n,l,t}v_{n,l}}\right\|_2}
\end{aligned}
$$
不同于用softmax生成attention，作者这里直接对向量积应用了ReLU。作者的解释我贴一下原话

> Indeed for irrelevant image-sentence pairs, the attention maps would be almost all zeros while the softmax process would always force attention to be a distribution over the image/words summing to 1. Furthermore, a group of words shaping a phrase could have the same attention area which is again hard to achieve considering the competition among regions/words in the case of applying softmax on the heatmap. 

对于不相关的Image-Sentence，softmax总会生成一个和为1的分布，然而每个word对Visual feature的attention都趋近于0才对，所以作者改用ReLU（这个问题论文**Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering**有更加灵活的解决方法，不过作者没有比对）。另外作者认为应用softmax会让同一个短语中的若干词具有类似的attention area，更难处理词/区域之间的竞争（没读懂）。

公式中的$a_{t,l}$表示了词$t$在图像特征中第$l$层的attended表示，作者还认为上述融合过程相当于从visual representation中用attention筛选出subset并构成了一个超平面，而$a_{t,l}$是其中一条向量。

##### Feature Level Selection

作者认为，word与图像的匹配性表现在与各layer的匹配性上，需要选择一个最匹配的feature level。
$$
\begin{aligned}
R_{t,l}&=\left<a_{t,l},s_t\right>\\
R_{t}&=\max_l{R_{t,l}}
\end{aligned}
$$
因为两者都是单位向量所以$R_{t,l}$既表示了余弦相似度也表示了投影值。作者把寻找最匹配level的过程形容成寻找最大投影超平面的过程。

> This procedure can be seen as finding projection of the textual embeddings on hyperplanes spanned by visual features from different levels and choosing the one that maximizes this projection. Intuitively, that chosen hyperplane can be a better representation for visual feature space attended by word t.

个人认为这个说法是有问题的，$R_{t,l}=\left<a_{t,l},s_t\right>$计算的仅仅是两个向量间的投影，作者没有也无法证明$a_{t,l},s_l$组成的平面能和作者所称的超平面相切，所以不能用$R_{t,l}$代替$s_l$到超平面的投影。不过撇开超平面的概念，这一套流程确实能找到与word最匹配的layer，$a_{t,l}$表示了词$t$在图像特征中第$l$层的attended表示。

最后作者分别用word-based/sentence-based相似度表示Image-Sentence的匹配分数：
$$
R_w\left(S,I\right)=\mathrm{log}{\left(sum^{T-1}_t{\mathrm{exp}\left(\gamma_1 R_t\right)}\right)^{\frac{1}{\gamma_1}}}
$$
$R_w\left(S,I\right)$是word-based相似度。
$$
\begin{aligned}
H_{n, l}^{s}&=\max \left(0,\left\langle\overline{\mathbf{s}}, \mathbf{v}_{n, l}\right\rangle\right)\\
\mathbf{a}_{l}^{s}&=\sum_{n=1}^{N} H_{n, l}^{s} \mathbf{v}_{n, l}\\
R_{s, l}&=\left\langle\mathbf{a}_{l}^{s}, \overline{\mathbf{s}}\right\rangle\\
R_{s}(S, I)&=\max _{l} R_{s, l}
\end{aligned}
$$
$R_{s, l}\left(S,I\right)$表示的是sentence-based相似度，其相当于把word-based的过程用整个句子的表示替换，所以直接用$\max_l R_{s,l}$就可以了。

##### Training

作者提出了一个很有意思的训练方式，对于每一个batch的image-caption pairs，对于image相当于要在batch中找到最佳的caption，反之同理，这就变成了一个分类任务。
$$
\begin{aligned}
P_x\left(S_b|I_b\right)&=\frac{\exp\left(\gamma_2 R_x\left(S_b,I_b\right)\right)}{\sum^B_{b'}\exp\left(\gamma_2 R_x\left(S_{b'},I_b\right)\right)}\\
P_x\left(I_b|S_b\right)&=\frac{\exp\left(\gamma_2 R_x\left(S_b,I_b\right)\right)}{\sum^B_{b'}\exp\left(\gamma_2 R_x\left(S_b,I_{b'}\right)\right)}\\
L^x&=-\sum^B_{b=1}\left(\log{P_x\left(S_b|I_b\right)}+\log{P_x\left(I_b|S_b\right)}\right)\\
L&=L^w+L^s
\end{aligned}
$$



### Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering

2018 CVPR

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

作者也借鉴了Transformer中multi-head attention的思路，引入$h$个head，每个head将特征映射到$d_h\left(\equiv d/h\right)$维空间再生成attention矩阵。
$$
\begin{aligned}
A_l^{\left(i\right)}&=\left(W_{V_l}^{\left(i\right)}V_l\right)^{\mathrm T}\left(W_{Q_l}^{\left(i\right)}Q_l\right)\\
A_{Q_l}&=\frac{1}{h}{\sum_{i=1}^h{\mathrm{softmax}\left(\frac{A_l^{\left(i\right)}}{\sqrt{d_h}}\right)}}\\
A_{V_l}&=\frac{1}{h}{\sum_{i=1}^h{\mathrm{softmax}\left(\frac{{A_l^{\left(i\right)}}^{\mathrm T}}{\sqrt{d_h}}\right)}}
\end{aligned}
$$
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

### Multi-task Learning of Hierarchical Vision-Language Representation

2019 CVPR

#### Keywords

多任务、Vision-Language表示学习

#### 解析

这篇文章是**Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering**作者的后续工作，其使用堆叠的Dense Co-Attention（上文提出的方法），对跨模态特征进行抽取和融合，同时作者设计了多任务结构，从堆叠而成的pipeline中延伸出分支，指向不同的任务，通过多任务指导特征表示学习。

![image-20200208202708938](imgs\image-20200208202708938.png)

##### Shared Encoder

作者使用了Dense Co-attention layer作为跨模态编码层，通过堆叠这些层以获得层次化的特征。每一层的输出可以表示为$\left(S_l,I_l\right)$中，$S_l\in \mathbb{R}^{d\times N},I_l\in \mathbb{R}^{d\times T}$，其中，$S_l,N$表示Sentence feature和词的数目，$I_l,T$表示Visual feature序列和regions的数目。

##### Task-specific Decoders

作者设计了一个Task-specific解码器，每个任务独享一个解码器，将$\left(S_l,I_l\right)$解码得到任务所需要的输出$O$，$l$是任务从编码器分支出来的层号。这里作者介绍了三个任务的做法：

Image Caption Retrival任务：

作者设计了两个具有相同结构的summary network把序列特征编码为表示整个图片/句子的特征。以图像为例，作者为每个region生成了$K$个分数，分别在$1...K$的分数上对regions做softmax，得到Attention weights，再合并$K$组weights得到最终关于各region的attention，通过加权平均的方式聚合。作者认为这比直接生成Attention weights能捕获到更多变的attention分布。
$$
\begin{aligned}
c^I_t&=\left[c^I_{t,1}...c^I_{t,K}\right]=\mathrm{MLP}_I\left(I_{l,t}\right)\\
\alpha^I_t&=\frac{1}{K}\sum^K_{k=1}\mathrm{softmax}\left(c^I_{1,k}...c^I_{T,k}\right)\\
v_I&=\sum^T_{t=1}{\alpha^I_t I_{l,t}}
\end{aligned}
$$
$c^I_t \in \mathbb{R}^K$是一个region $t$的$K$个分数，$\alpha^I_t$是其最终的Attention weight。$v_I$是聚合的图像特征，文本同理。匹配得分可以表示为
$$
\mathrm{score}\left(I,S\right)=\sigma\left({v_I}^{\mathrm{T}}W v_S\right)
$$
$W$统一两者的特征空间，$\sigma$是logistic function(sigmoid)。

VQA任务：

作者用相似的方法进行特征融合，答案选取有两种类型：

* 候选答案集固定：$\mathrm{score}\left(I,S\right)=\sigma\left(\mathrm{MLP}\left[v_I\oplus v_S\right]\right)$，$\oplus$表示拼接或相加。
* 候选集不固定：$\mathrm{score}\left(S_a,I,S\right)=\sigma\left({S_a}^{\mathrm{T}}W \left(v_I+v_S\right)\right)$。其中$S_a$表示答案$a$的得分。

Visual Grounding任务：

需要对齐句子中的phrase和图像中的region，pharse的特征可以从Sentence中分离出来 $p_h=\mathrm{AvgPooling}\left(S\left[b_h:e_h\right]\right)$，其中$b_h,e_h$表示了第$h$个短语的起始位置，通过短语各单词表示的均值池化得到短语表示。
$$
\mathrm{score}\left(I_t,p_h\right)=\sigma\left({p_h}^{\mathrm{T}}W I_t\right)
$$

##### Task-specific Hyperparameter Search and Schedule

对于每个任务，其分支层数$l$及其他任务相关参数的选择，作者首先在训练独立任务时执行了grid search，确定超参数后再联合训练。

由于不同任务需要的训练步数和参数不同，作者提出两种训练方法：

* Curriculum Learning，从一个任务开始，逐步增加每一轮迭代参与的任务数目。
* Periodical task switch，首先为各任务设计迭代次数$C\alpha_i$（$C$认为指定，各任务共享，$\alpha_i$由超参数搜索得到）。在每轮循环中，按顺序训练Task，并迭代任务指定的次数，训练完所有任务后进入下一轮循环，直到达到人工预设的循环上限。

### Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment 

2019 ICCV

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

### CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval 

2019 ICCV

#### Keywords

跨模态、Text-Image检索、鲁棒性

#### 解析

这篇文章吹牛和抄的内容比较多，就捡一些有贡献的地方写吧。

其提出先有很多方法在做跨模态融合时都是基于Image-Text是相匹配的前提，但训练时负样本实际上是不匹配的，这会误导模型训练，所以作者设计了一个Gated Fusion方法，希望在Text与Image不匹配时，能各自保持自己原本的特征。

为了保持行文完整性，还是写一下整个论文的步骤。

##### Cross-modal Message Aggregation

作者说这个部分是**we propose**的，然而这里的方法完全抄袭2018 CVPR Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering（笔记也在本页面），也没有在正文中出现对它的引用！

它会基于Co-Attention，为Text中每个词生成来自Visual的Attended表示，同理每个Region生成来自Text的表示，记为$\tilde{T},\tilde{V}$。

##### Cross-modal Gated Fusion

作者希望对于不匹配的Text和Image，不要将Attended representation（来自另一模态的特征）传递过去。

对于视觉表示
$$
\begin{aligned}
G_v&=\sigma\left(V\odot \tilde{T}\right)\\
\hat{V}&=\mathcal{F_v}\left(G_v\odot\left(V\oplus \tilde{T}\right)\right)+V
\end{aligned}
$$
如果attend feature和原本的特征不匹配，那么Gate值就低，则会跟多保留原特征。文本同理，不再赘述。

...

##### Loss

作者提出了一个有趣的Loss，基于Rank loss做了改进

* 在一个mini-batch中，只计算最困难的负样本的loss
* 将基于相似度的Loss改为基于二元交叉熵的Loss

$$
\mathcal{L}_{BCE\text{-}h}\left(I,C\right)=\underbrace{\log\left(m\left(I,C\right)\right)+\max_{C'}\left[\log\left(1-m\left(I,C'\right)\right)\right]}_{image-to-text\,matching\,loss}\\
+\underbrace{\log\left(m\left(I,C\right)\right)+\max_{I'}\left[\log\left(1-m\left(I',C\right)\right)\right]}_{text-to-image\,matching\,loss}
$$

$m$表示模型。





