# Visual Language Tasks Survey

[TOC]

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

### Unsupervised Image Captioning

2019 CVPR

**Keywords:** 无监督学习、Image Caption

本文是近年第一个讨论无监督训练Image Caption任务的文章。这一任务可以看作数据集中有大量图像和语料，但语料和图像之间没有任何对应关系。

如何从无监督数据中学习到描述指定图像的文本，作者提出了一个大体的思路，其认为这个任务所需要的知识可以分为两个部分

* 如何从图像中获取具有语义的concepts
* 如何将concepts转化为自然语言

前者作者主要使用从现有的detector中蒸馏知识，同时通过reconstrust image的过程加以辅助，使得模型能抽取出图像中具有丰富语义的中间表示。后者作者设计了一个自学习机制，从语料中抽取concepts并将原句子作为ground truth以进行训练。

#### 模型结构

核心的模型结构和大多数Caption模型无异，首先由一个CNN网络从Image中抽取中间表示，再通过一个基于LSTM的generator生成Caption语句。
$$
\begin{aligned}
f_{im}&=\mathrm{CNN}\left(I\right)\\
x_{-1}&=\mathrm{FC}\left(f_{im}\right)\\
x_t&=W_es_t\\
\left[p_{t+1},h^g_{t+1}\right]&=\mathrm{LSTM}^g\left(x_t,h^g_t\right)\\
s_t&\sim p_t\quad t\in \{1...n\}
\end{aligned}
$$
LSTM的每一步输入$x_t$为上一轮产生单词的embedding，$W_e$是embedding字典，图像特征通过变换到词的向量空间中作为LSTM的初始输出。其每一轮会输出一个词汇上的概率分布，通过采样的方式选择每一步的输出词$s_t$。

为了能让模型能被优化，作者设计了一个discriminator用来判别句子的真实性。其同样是基于LSTM的结构。
$$
\left[q_{t},h^d_{t}\right]=\mathrm{LSTM}^d\left(x_t,h^d_t\right)
$$
其接收geneator生成句子的embedding序列，每一步输出当前形成句子的置信度。

![image-20200221191001278](imgs\image-20200221191001278.png)

#### 监督信号构成

本文的监督信号由两类组成，Loss通过梯度下降的方式进行传播，Reward则通过Policy Gradient的方式模拟梯度。

##### 对抗监督

作者设计的discriminator可以和generator构成GAN模型，但值得注意的是，生成器和判别器之间由于有采样的存在，无法传递梯度，因此作者将判别器的结果构造成adversarial reward。
$$
r^{adv}_t=\log\left(q_t\right)
$$
同时判别器本身是可以通过基于真实语料构造的数据产生loss进行训练的。通过真实语料，可以构造带有标注的真实和虚假语料库，将判别器单独作为分类器进行训练。
$$
\mathcal{L}_{adv}=-\left[\frac{1}{l}\sum^l_{t=1}{\log\left(\hat{q}_t\right)}+\frac{1}{n}\sum^n_{t=1}{\log\left(1-q_t\right)}\right]
$$

##### 视觉概念蒸馏

上述对抗监督只能指导生成器通过image feature生成通顺的句子，但并没有和图像中的语义建立联系。为此作者希望将现有的detector的知识蒸馏到生成器中，具体而言，检测器检测到的concepts可以表示为$\mathcal{C}=\{\left(c_1,v_1\right),...,\left(c_{N_c},v_{N_c}\right)\}$，其中$c$表示concept，$v$表示置信度，如果生成器产生的句子中存在检测器检测到的concept就给予reward。
$$
r^c_t=\sum^{N_c}_{i=1}I\left(s_t=c_i\right)v_i
$$

##### 图像文本重构

检测器只能提供少量精确的语义指导，现有的检测器能够检测的物体数目过少难以让模型理解广泛的语义。为此作者提出让sentences和images的特征表示映射到同一空间中，并反向重构两者。

**图像重构：**考虑到文本本身并不会对图像的细节进行描述，所以作者考虑仅还原图像抽取后的特征向量。
$$
\begin{aligned}
x'&=\mathrm{FC}\left(h^d_n\right)\\
\mathcal{L}_{im}&=\lVert x_{-1}-x'\rVert^2_2\\
r^{im}_t&=-\mathcal{L}_{im}
\end{aligned}
$$
其中loss可以用来训练判别器，而reward用来训练生成器。

**文本重构：**如果调换生成器和判别器的位置，可以把它们看作是Encoder-Decoder模型，判别器起到了Encoder的作用输入句子输出的hidden state可以看作sentence feature，而生成器读取image feature生成句子，起到了Decoder的作用，那么只要让这两种feature处于同一特征空间，Encoder-Decoder模型就构建起来了。为此，作者将判别器的输入替换成语料库并要求模型重构句子自身。为了贴近从image feature中解码句子的场景，作者引入了文本去噪自编码器的思路，在输入的句子中引入噪声，要求复原无噪声的句子，把图像特征解码句子看作是文本去噪的过程。
$$
\mathcal{L}_{sen}=-\sum^l_{l=1}\log\left(p\left(s_t=\hat{s}_t|\hat{s}_1,...,\hat{s}_{t-1}\right)\right)
$$
调换位置得到的Encoder-Decoder模型是梯度连续的，可以直接用loss训练生成器和判别器。

![image-20200221191143233](imgs\image-20200221191143233.png)

#### 梯度整合

最终将Policy Gradient和Loss Gradient结合，生成器的梯度表示为
$$
\begin{aligned}
\nabla_{\theta} \mathcal{L}\left(\theta\right)=&-\mathbb{E}\left[\sum^n_t{\left(\sum^n_{s=t}\gamma^s\left(r^{adv}_s+\lambda_c r^c_s\right)+\lambda_{im} r^{im}_s-b_t\right)}\nabla_{\theta}\log\left(s_t^{\mathrm{T}}p_t\right)\right]\\
&+\lambda_{sen}\nabla_{\theta}\mathcal{L}_{sen}\left(\theta\right) 
\end{aligned}
$$
判别器的梯度均来自梯度下降，Loss表示为
$$
\mathcal{L}_D=\mathcal{L}_{adv}+\lambda_{im}\mathcal{L}_{im}
$$
其中$\lambda$均为均衡各模块的超参数，在Policy Gradient中$\gamma$是衰减参数，$b$是baseline  reward估计参数。

### Visual Concept-Metaconcept Learning

2019 NIPS

**Keywords:** Visual concept recognition

在建立视觉和concept的联系中，现有方法往往时孤立地考虑每一个concept于其视觉特征，作者考虑让计算机理解哪一些concept描述的是同一类事物，并将其称为metaconcept。

![image-20200219203726608](imgs\image-20200219203726608.png)

在上图中，如果存在bias的数据，比如red的物体大多是cube，那么模型很可能将red这一概念和视觉特征cube联系到一起，但如果模型能理解red和green是描述物体的同一类属性，同时由于视觉表示上，色彩和形状特征差异较大，那么模型就可以纠正这一bias，对red的正确认知还可以泛化到未见过的组合中比如red cylinder。同时如果模型能学会判别相同类型的concept，那么也可能泛化到理解sphere和cube是描述形状的concept。

基于如上需求，作者基于神经符号推理，拓展出了Meta Verify机制。

![image-20200219204818915](imgs\image-20200219204818915.png)

a b c.I部分都在论文**The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision**中有所讲解，其将物体从视觉图像中检测出，同时将问题解析为可执行程序，在神经符号推理中执行并求解答案。作者的贡献主要是c.II中的Meta Verify。

首先作者提出了Metaconcept questions，这是一种纯文本问题，问题会询问一对oncepts之间的metaconcept relation，比如red与green; cube与sphere都属于same kind这一metaconcept，将这些输入模型，需要给出其检验得到的置信度。

为了处理concept之间的关系，作者提出了一个half-space定义规则$V\left(x\right)=\left\{y\in\mathbb{R}^N|\left(y-x\right)^{\mathrm{T}}x>0\right\}$。作者假设整个空间服从标准正态分布，那么概念$a$的概率表示为
$$
\begin{aligned}
\mathrm{Pr}\left(a\right)&=\mathrm{Vol}_{\mathcal{N}\left(0,I\right)}\left(V_a\right)=\int_{z\in V_a}{\frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}\lVert z\rVert^2_2}dz}=\frac{1}{2}\left[1-\mathrm{erf\left(\frac{\lVert x\rVert_2}{\sqrt{2}}\right)}\right]\\
\mathrm{Pr}\left(a,b\right)&=\mathrm{Vol}_{\mathcal{N}\left(0,I\right)}\left(V_a\cap V_b\right)\\
\mathrm{Pr}\left(b|a\right)&=\frac{\mathrm{Pr}\left(a,b\right)}{\mathrm{Pr}\left(b\right)}=\frac{\mathrm{Vol}_{\mathcal{N}\left(0,I\right)}\left(V_a\cap V_b\right)}{\mathrm{Vol}_{\mathcal{N}\left(0,I\right)}\left(V_a\right)}
\end{aligned}
$$
其中$\mathrm{erf}\left(\right)$表示**error function**。

![image-20200219205843176](imgs\image-20200219205843176.png)

以red为例，图中红色区域既是其定义的half-space，那么object is red可以表示为$\mathrm{Pr}\left(red|o\right)$。

定义了概率表示后，对于任意一种metaconcept（如same kind或synonym），作者会给出一个专属的多层感知器$f_{synonym}$，作者进而基于概率表示计算了concept之间的关系线索


$$
\begin{aligned}
g_1\left(a,b\right)&=\mathrm{logit}\left(\mathrm{Pr}\left(a|b\right)\right)\\
g_2\left(a,b\right)&=\mathrm{ln}\frac{\mathrm{Pr}\left(a,b\right)}{\mathrm{Pr}\left(a\right)\mathrm{Pr}\left(b\right)}\\
\end{aligned}
$$

$$
\begin{aligned}
&\mathrm{MetaVerify}\left(red,cube,synonym\right)=\\
&\quad\quad f_{synonym}\left(g_1\left(red,cube\right),g_1\left(cube,red\right),g_2\left(red,cube\right)\right)
\end{aligned}
$$

其输出核验的置信度。

### Detecting Unseen Visual Relations Using Analogies 

2019 ICCV

**Keywords:** Visual concept recognition

本文基于迁移的思想，让模型能够识别未见过的视觉概念组合。即对于$t=\left(s,p,o\right)$这种基于subject,predicate和object的三元组，如果其中每个元素都得到了充分的学习但$t$未曾参与过训练，希望识别到$t$和视觉表示的对应。

视觉概念学习的传统方法有两种思路

* 为每个entity分别学习检测器，并用合并后的检测器结果作为对应，对于subject和object，检测器是容易学习的，但对于谓词predicate，其在视觉空间中存在较大的variability，很难学习。
* 为每种组合学习检测器，显然其数据要求是巨大的，也无法处理unseen combination。

作者提出了归类迁移的思想，既学习细分的entity的表示，也构建整个visual phrase的表示。

![image-20200219215706716](imgs\image-20200219215706716.png)

在训练过程，作者分别抽取subject/object/predicate/visual phrase的视觉和三元组特征，并分别嵌入到四个特征空间中。图像与短语间的整体对应和训练Loss表示如下
$$
\begin{aligned}
S_{t,i}&=\prod_{b\in \left\{s,p,o,vp\right\}}{\frac{1}{1+e^{-{w^b_t}^{\mathrm{T}}v^b_i}}}\\
\mathcal{L}_b&=\sum^N_{i=1}{\sum_{t\in V_b}{\left[y^i_t=1\right]\log\left(\frac{1}{1+e^{-{w^b_t}^{\mathrm{T}}v^b_i}}\right)}}\\
\mathcal{L}&=\mathcal{L}_s+\mathcal{L}_o+\mathcal{L}_p+\mathcal{L}_{vp}
\end{aligned}
$$
对于未见过的组合$t'$，作者考虑将已有的三元组表示进行变换得到未见过的三元组表示$w^{vp}_{t'}$，就像word2vec中的经典例子**"king"-"man"+"woman"="queen"**，作者希望达到**"person ride hourse"-"hourse"+"cow"=person ride cow"**这种效果。为此作者设计了如下函数
$$
\begin{aligned}
w^{vp}_{t'}&=w^{vp}_{t}+\Gamma\left(t,t'\right)\\
\Gamma\left(t,t'\right)&=\mathrm{MLP}\left[
\begin{smallmatrix}
      w^{s}_{t'}-w^{s}_{t}\\
      w^{o}_{t'}-w^{o}_{t}\\
      w^{p}_{t'}-w^{p}_{t}
    \end{smallmatrix}\right]
\end{aligned}
$$
为了提高鲁棒性，作者设计相关函数并整合多个三元组迁移的结果
$$
\begin{aligned}
G\left(t,t'\right)&=\sum_{b\in \left\{s,p,o,vp\right\}}{\alpha_b {w^b_t}^{\mathrm{T}}w^b_{t'}}\\
\bar{w}^{vp}_{t'}&=\sum_{t \in \mathcal{N}_{t'}}{G\left(t,t'\right)\left(w^{vp}_{t}+\Gamma\left(t,t'\right)\right)}
\end{aligned}
$$

### DenseCap: Fully Convolutional Localization Networks for Dense Captioning

2016 CVPR

**Keywords:** Localization

本文的任务是从图像中生成Dense Caption（图像中有多个目标，需要给出多条语句描述目标和目标之间的关系），作者主要的贡献是在Localization中将Faster-RCNN中的RoI Pooling替换成bilinear interpolation，使得梯度传播中可以保持空间位置。

![image-20200219194707797](imgs\image-20200219194707797.png)

传统的Faster-RCNN会使用RoI Pooling的方式将Region划分成由大小不一的cells构成的固定大小的grid，通过对cell取max pooling，得到固定大小的feature map。这种方式可以实现梯度的反向传播，但梯度并不是直接返回输入的各个坐标(max pooling反向传播时只会将梯度直接传给响应max的位置)。

作者提出使用bilinear interpolation予以替换。

对于任意尺寸$C\times W\times H$的feature map $U$，需要得到$C\times X\times Y$的输出$V$，在此之前，可以计算出$V$中任意坐标到原始输入$U$的映射，这个映射可以存入Sampling Grid $G\in \mathbb{R}^{X\times Y\times 2}$中其中$G_{i,j}=\left(x_{i,j},y_{i,j}\right)$，表示$V$中 $i,j$ 像素在 $U$ 中的原坐标。随后便可以基于$G$和预设的kernel计算 $V$ 了
$$
\begin{aligned}
V_{c,i,j}&=\sum^W_{i'=1}{\sum^H_{j'=1}{U_{c,i'j'}k\left(i'-x_{i,j}\right)k\left(j'-y_{i,j}\right)}}\\
k\left(d\right)&=\max\left(0,1-\lvert d\rvert\right)
\end{aligned}
$$
如果有 $B$ 个Region就重复上述操作并将 $B$ 个regions的结果堆叠就得到了最终的feature tensor of shape $B\times C\times X\times Y$。

本文其他地方都比较老套，就不进一步解释了。

### Auto-Encoding Scene Graphs for Image Captioning

2019 CVPR

**Keywords:** GCN、跨模态、自编码器

关于Caption生成任务，一直以来存在一个问题就是语言缺少归纳性，把目标检测的结果拼凑得到文本语义往往比较简单，而人类在描述时会基于一些预先学习的假设(inductive bias)，得到高度概括的语句。因此作者提出一种图卷积+自编码器的方法挖掘更深层次的语义并进行归纳总结，以提升Caption的效果。

#### Encoder-Decoder

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

#### Auto-Encoding Scene Graphs

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

#### Multi-modal Graph Convolution Network

Image2Caption部分的视觉特征$\mathcal{V}$同样可以生成Graph，作者将文本模态(label embdding $e_{o_i}$)和视觉模特(visual embedding $v_{o_i}$)生成Graph的节点编码进行融合

$u_{o_i}=\mathrm{ReLU}\left(W_1 e_{o_i}+W_2 v_{o_i}\right)-\left(W_1 e_{o_i}+W_2 v_{o_i}\right)^2$

再使用与SGAE相同的方式生成Sentence。

### Parallel Attention: A Unified Framework for Visual Object Discovery through Dialogs and Queries

2018 CVPR

**Keywords:** Attention、Visual Grounding、Object Recognize

本文针对的任务是Visual Grounding任务，给出一张图片和一些有关文本，需要找出其匹配的图像区域。

作者提出了一个基于对话和询问的目标认知模型，提出了一种不需要固定标签集合的目标认知方法。其认为模型在识别目标时，首先应该关注到大体的候选区域，在读取自然语言输入时一步步缩小范围，最终收敛到一个局部区域。

为此作者提出了一个two-way attention mechanism

* Image-level attention: 建立文本上下文信息与图片的联系（如场景）
* Propose-level attention: 建立候选区域与自然语言中关键词的联系。

![image-20200207204239923](imgs\image-20200207204239923.png)

注：自然语言输入有两种，一种是问答序列，另一种是一个句子，在图片中，$d_1,...d_L$并不是说模型需要同时接收这两者，而是当输入为问答序列时，每次读取一个问答对并走橙色路线，反之每次读取一个单词，走蓝色路线。

#### Image-level attention

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

#### Propose-level attention

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

#### Referring

作者通过让每个候选region的最终表示$\tilde{p}_i$和image-level attention的表示进行点积，并通过softmax得到区域选中的概率分布。使用交叉熵损失训练。
$$
P=\mathrm{softmax}\left(h_{t=L}\odot\tilde{P}\right)
$$
这一方法并不需要人为标注目标是什么，但需要给出候选区域（作者使用交叉熵而非IoU）标注，而且多轮对话实际运用中也没那么理想。

### Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering

2018 CVPR

**Keywords:** VQA、Image Caption、Top-Down Attention

这篇文章透露出浓浓的竞赛风格，很多细节为什么要这么做的解释比较含糊，一些概念和叫法也略显强行（比如Bottom-Up Attention），不过这也正常，很多东西就是解释不了的，效果好学就行了。

这篇文章提出了一个基于Attention的模型为Image Caption和VQA生成融合多模态的Attended feature，其包含两类Attention：

* Bottom-Up Attention：相比于直接用CNN抽取特征，作者改用Faster-RCNN从图片中取出图像区域并抽取特征。这可以看作是一种Hard Attention，充分训练的网络可以直接筛选出可能有用的图像区域。因为是从像素生成Region，所以是Bottom-Up。
* Top-Down Attention：作者考虑了Caption任务和VQA任务的Top-Down Attention，其思想便是用整体的文本表示指导个Regions的关注度，所以是Top-Down。

#### Bottom-Up Attention

Faster RCNN就不过多介绍了，作者在ImageNet预训练模型上进一步用Visual Genome上训练，这个数据集还保留了物体的attribute特征，所以作者额外引入了属性分类任务进行训练。

#### Top-Down Attention

作者考虑了Caption任务和VQA任务。

Caption任务：

作者使用了两组具有相同结构的LSTM进行编码$h_t=\mathrm{LSTM}\left(x_t,h_{t-1}\right)$。Top-Down Attention LSTM标记为$\mathrm{LSTM_1}$，Language LSTM标记为$\mathrm{LSTM_2}$。

![image-20200209171950282](imgs\image-20200209171950282.png)

Top-Down Attention LSTM的输入是将Language LSTM上一步的输出和regions的特征均值以及当前的输入词嵌入（这里作者没说这个输入词是什么，从其引用文献推测应该是上一步预测的词）进行拼接得到。
$$
x^1_t=\left[h^2_{t-1},\mathrm{mean}\left(v_1...v_k\right),W_e\Pi_t\right]
$$
其输出被用于生成关于Regions的attention map并得到关于整个图片的attended feature。
$$
\begin{aligned}
a_{i,t}&=w^{\mathrm{T}}_a \mathrm{tanh}\left(W_{va}v_i+W_{ha}h^1_t\right)\\
\alpha_t&=\mathrm{softmax}\left(a_{1:k,t}\right)\\
\hat{v}_t&=\sum^K_{i=1}\alpha_{i,t}v_i
\end{aligned}
$$
Language LSTM的输入来自于Top-Down Attention LSTM的当前隐状态和图片的attended feature。
$$
x^2_t=\left[\hat{v}_t,h^1_t\right]
$$
每一个时间步，其隐藏状态会用于生成一个单词
$$
\begin{aligned}
p\left(y_t|y_{1:t-1}\right)&=\mathrm{softmax}\left(W_ph^2_t+b_p\right)\\
p\left(y_{1:t}\right)&=\prod^T_{t=1}{p\left(y_t|y_{1:t-1}\right)}
\end{aligned}
$$
作者使用了交叉熵和CIDEr损失进行训练，后者基于强化学习可以模拟梯度。

VQA任务：

![image-20200209202321716](imgs\image-20200209202321716.png)

这里的Top-Down Attention和上文类似，只不过其输入单词换成问句，其最后一个隐藏层的输出作为问题的表示$q$，此外为了简化表述，作者使用了候选答案集固定的VQA模式，模型需要输出候选答案的概率分布。

作者设计了一个非线性变化$y=f\left(x\right)$，以提高非线性拟合能力。
$$
\begin{aligned}
\tilde{y}&=\mathrm{tanh}\left({Wx+b}\right)\\
g&=\sigma\left(W'x+b'\right)\\
y&=\tilde{y}\circ g
\end{aligned}
$$
类似于Caption任务，作者生成question-attended的视觉信息，并与文本问题的表示融合，进而生成答案分布
$$
\begin{aligned}
a_i&=w^{\mathrm{T}}_a f_a\left(\left[v_i,q\right]\right)\\
\hat{v}&=\sum^K_{i=1}{a_iv_i}\\
h&=f_q\left(q\right)\circ f_v\left(\hat{v}\right)\\
p\left(y\right)&=\sigma\left(W_of_o\left(h\right)\right)
\end{aligned}
$$

### Visual Semantic Reasoning for Image-Text Matching

2019 ICCV

**Keywords:** Text-Image Matching、视觉推理、弱监督

本文应用的任务也是图像和文本的匹配任务，作者考虑到视觉图像中除了Object和Sense外，它们之间的interaction, relative positions等high-level的语义没有没很好地考虑到，所以作者提出了一个视觉表示学习模型，其能够捕获到Objects和它们的语义关系。

本文的融合了GCN做局部推理以及LSTM做局部特征的融合（也被称作全局推理）这两种思路。

![image-20200208174610546](imgs\image-20200208174610546.png)

#### Image Representation by Bottom-Up Attention

作者首先通过Bottom-Up Attention方法从图像$I$中获取region representations $V=\left\{v_1,...,v_k\right\}$。

Bottom-Up Attention是2018 CVPR的Oral，本文也会对其进行讲解。

#### Region Relationship Reasoning

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

#### Global Semantic Reasoning

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

####  Learning Alignments by Joint Matching and Generation

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

**Keywords:** Layer Attention、跨模态

这篇文章提出了一个新视角，将文本对视觉的注意力应用在视觉特征抽取的过程中，而非仅对抽取结果进行Attention。

#### Multi-Level Multimodal Attention Mechanism

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

#### Feature Level Selection

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

#### Training

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

**Keywords:** Attention、跨模态表示学习

作者基于自然语言处理中的Transformer模型，提出了一种跨模态的Co-Attention模型，以图片和其描述为例，抽取出Regions和Phrases序列，可以分别生成每个Phrase对所有Regions的attended feature和Region对所有Phrases的attended feature。

为了表述方便，这里用$Q_l$指代words feature序列并用$V_l$指代pixel feature(经过卷积抽取过的)序列，作者堆叠了Co-attention并用$l$表示层数。

#### Co-attention

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

**Keywords:** 多任务、Vision-Language表示学习

这篇文章是**Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering**作者的后续工作，其使用堆叠的Dense Co-Attention（上文提出的方法），对跨模态特征进行抽取和融合，同时作者设计了多任务结构，从堆叠而成的pipeline中延伸出分支，指向不同的任务，通过多任务指导特征表示学习。

![image-20200208202708938](imgs\image-20200208202708938.png)

#### Shared Encoder

作者使用了Dense Co-attention layer作为跨模态编码层，通过堆叠这些层以获得层次化的特征。每一层的输出可以表示为$\left(S_l,I_l\right)$中，$S_l\in \mathbb{R}^{d\times N},I_l\in \mathbb{R}^{d\times T}$，其中，$S_l,N$表示Sentence feature和词的数目，$I_l,T$表示Visual feature序列和regions的数目。

#### Task-specific Decoders

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

#### Task-specific Hyperparameter Search and Schedule

对于每个任务，其分支层数$l$及其他任务相关参数的选择，作者首先在训练独立任务时执行了grid search，确定超参数后再联合训练。

由于不同任务需要的训练步数和参数不同，作者提出两种训练方法：

* Curriculum Learning，从一个任务开始，逐步增加每一轮迭代参与的任务数目。
* Periodical task switch，首先为各任务设计迭代次数$C\alpha_i$（$C$认为指定，各任务共享，$\alpha_i$由超参数搜索得到）。在每轮循环中，按顺序训练Task，并迭代任务指定的次数，训练完所有任务后进入下一轮循环，直到达到人工预设的循环上限。

### Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment 

2019 ICCV

**Keywords:** Visual grounding、Image Level监督

本文研究的是Visual grounding任务，使用Image level的监督数据，实现Image regions和Caption phrases之间的对齐。其提出早期方法直接用局部对齐的置信度来生产全局对齐的置信度，模型只需要学习判别一些关键区域的对齐就足以实现Image和Caption的对齐，这使得RoIs和短语的对齐训练不充分。作者考虑将模型认为对齐的部分抽取出来，仅使用对齐部分的feature作为输入，让Image Caption的对齐判断仅依赖于对齐的输入，这就要求网络需要先获取比较精细的对齐结果，这更加充分地利用了Image-Level的监督信号。

作者首先通过预训练网络获取Region of interests(RoIs)和Caption phrases和对应的特征向量，并进行使用过Image level监督实现这两者的对齐。

为此作者提出了三个模块：

* The Local Matching Module:对于给定的Caption请求，推断所有潜在的RoI-phrase对应关系
* The Local Aggregator Module:为上一模块的每一条对齐结果，生成caption-conditioned的image representation。
* The Global Matching Module:使用caption-conditioned的image representation判别与Caption的匹配程度。

![image-20200205214846947](imgs\image-20200205214846947.png)

#### The Local Matching Module

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

#### The Local Aggregator Module

对于图像I，其基于caption c的RoIs可以表示为$I^c_{rois}=\left(x^c_k\right)^P_{k=1}$，需要基于它获取caption-conditioned的image representation，作者直接用MLP+mean pooling的方式编码$I^c_{rois}$，写作$f_{enc}\left(I^c_{rois}\right)$。

#### The Global Matching Module

最终作者直接用RNN编码的caption representation和caption-conditioned的image representation进行余弦相似度比较得到相似度，训练时使用Rank Loss。

### CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval 

2019 ICCV

**Keywords**: 跨模态、Text-Image检索、鲁棒性

这篇文章吹牛和抄的内容比较多，就捡一些有贡献的地方写吧。

其提出先有很多方法在做跨模态融合时都是基于Image-Text是相匹配的前提，但训练时负样本实际上是不匹配的，这会误导模型训练，所以作者设计了一个Gated Fusion方法，希望在Text与Image不匹配时，能各自保持自己原本的特征。

为了保持行文完整性，还是写一下整个论文的步骤。

#### Cross-modal Message Aggregation

作者说这个部分是**we propose**的，然而这里的方法完全抄袭2018 CVPR Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering（笔记也在本页面），也没有在正文中出现对它的引用！

它会基于Co-Attention，为Text中每个词生成来自Visual的Attended表示，同理每个Region生成来自Text的表示，记为$\tilde{T},\tilde{V}$。

#### Cross-modal Gated Fusion

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

#### Loss

作者提出了一个有趣的Loss，基于Rank loss做了改进

* 在一个mini-batch中，只计算最困难的负样本的loss
* 将基于相似度的Loss改为基于二元交叉熵的Loss

$$
\mathcal{L}_{BCE\text{-}h}\left(I,C\right)=\underbrace{\log\left(m\left(I,C\right)\right)+\max_{C'}\left[\log\left(1-m\left(I,C'\right)\right)\right]}_{image-to-text\,matching\,loss}\\
+\underbrace{\log\left(m\left(I,C\right)\right)+\max_{I'}\left[\log\left(1-m\left(I',C\right)\right)\right]}_{text-to-image\,matching\,loss}
$$

$m$表示模型。





