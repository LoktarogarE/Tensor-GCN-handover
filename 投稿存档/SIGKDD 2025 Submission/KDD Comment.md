# KDD Comment Stage 杂项

In section 2, in begins by noting "input data \bf X", and talks about the Laplacian. But without stating that there is a kNN graph construction there, the reader assumes that the Laplacian is defined over the full pairwise distance matrix, and if there is a graph built, it should be a complete graph. The kNN is not introduced (and even never defined) before section 2.3. The use of Fourier terms here is also misleading. Indeed Fourier transform is a specific example of linear space decomposition named "Spectral methods", but we are not using here a Fourier transform, with the frequency interpretations. A better reference to spectral methods in graph is needed. For example the well-known "Ulrike Luxburg. 2007. A tutorial on spectral clustering. Statistics and Computing 17, 4 (December 2007), 395–416. "。 I am not sure that you have addressed all notations issues. The issue is not related only to giving a reference to Von Luxburg paper. 

我的思路：我们并非只是添加了文献引用，而是在手稿中做出了大量修改，从谱分析的出发引出谱图方法，并在提出模型前讨论了，利用 knn 构建相似度图，特别是讨论相似度图的构建方式以及引入 kNN 的动机？





Thank you for your thoughtful reminder. We apologize for this and would be happy to provide additional details if the revewers would like clarification.

问题 1：I am not sure that you have addressed all notations issues. The issue is not related only to giving a reference to Von Luxburg paper. 

1. We have made multiple adjustments to Section 2 to eliminate ambiguity: 
   * We moved the k‑nearest‑neighbor graph procedure to beginning of Section 2.1, immediately after introducing the data matrix X.  There we define how we compute pairwise distances, select the top‑$k$ neighbors per node, apply the Gaussian kernel, and assemble the sparse similarity matrix $S$.  The (normalized) graph Laplacian L is now explicitly tied to that kNN graph.
   * In Section 2.2, we no longer invoke “Fourier transform” language; instead we refer generically to spectral decomposition of $L$ (its eigendecomposition) and how polynomial filters in that domain yield graph convolutions.

问题2：By using the diffusion framework as you have done in the W2-2, the diffusion do not only depend on direct neighbours but on all the graph (the function g can defined over the whole graph, in this context you do not need anymore to use the L3 tensor. As said in my initial review, tensor are only useful to simplify the calculations over GPU type of architecture and do not provide new insights. I am not seeing something in the paper that will really benefit theoretically from Tensor notation. The use of tensor in the title is therefore questionable.

回复：



Thank you for your thoughtful reminder. We apologize for this and would be happy to provide additional details if the revewers would like clarification.

1. We have made multiple adjustments to Section 2 to eliminate ambiguity: 

   * We moved the k‑nearest‑neighbor graph procedure to beginning of Section 2.1, immediately after introducing the data matrix X.  There we define how we compute pairwise distances, select the top‑k neighbors per node, apply the Gaussian kernel, and assemble the sparse similarity matrix S.  The (normalized) graph Laplacian L is now explicitly tied to that kNN graph.

   * In Section 2.2, we no longer invoke “Fourier transform” language; instead we refer generically to spectral decomposition of L (its eigendecomposition) and how polynomial filters in that domain yield graph convolutions.

2. It seems there might be some misunderstanding regarding our use of the $L_3$ tensor in Tensor‑GCN:

   - First we would like to clarify that the term "tensor" in our paper does not refer to GPU-oriented tensor computation, but rather to a higher-order affinity structure used for modeling triplet-level spatial relationships among samples. We invite you to consult "**General Response (Motivation and Contributions)** for an in‑depth discussion of the method’s motivation and the design rationale underlying our tensor similarity measure.
   - While a spectral filter g(L) indeed propagates information across the entire graph, it is fundamentally grounded in the pairwise Laplacian L, which under HDLSS conditions suffers from metric collapse and noise.  Thus, g(L) alone cannot faithfully capture true sample relationships, motivating our introduction of a complementary, high‑order tensor similarity.

   - The L_3 tensor defines a separate diffusion pathway that is **not** derived from the k‑NN graph structure (of L).  Instead, it encodes spatial relationships among triples of points.  During the convolution process, L and L_3 provide complementary information, jointly yield richer embeddings than either alone.

   We hope this clarifies why the L_3 tensor is both theoretically meaningful and indispensable in our model. Thank you again for helping us improve the manuscript!

   

   

   

   - Our choice of tensor notation is not merely to leverage GPU efficiency, but to formalize and integrate multi‑scale affinities (pairwise and triplewise) within a unified graph‑convolution framework.  This multi‑order fusion is critical for overcoming the limitations of traditional spectral methods on HDLSS data.





您可能误解了本文中对于 Tensor 的使用，我们为您 L3 在 Tensor-GCN 方法中的重要性：

* 从图扩散的角度角度看，g 的确能聚合整张图的结构信息，但扩散本身的依据：拉普拉斯矩阵L，由于高维环境下的度量塌陷，无法准确刻画样本间的关系，这是我们引入 张量建模样本关系的初衷。

* 在执行图扩散时，L3 并不受到 L2 的影响，因为它本质并不受 knn 图的结构信息指导，而是基于样本间的空间关系，对于样本的一种高阶图采样，代表着与 L2 完全不同的结构信息（相当于是两条独立的扩散通道，这里或许应该说不是一个kernel，这个我不确定）

* 本文中采用的张量并非是为了加速计算，考虑到 knn 图在分析高维数据中的缺陷，引入 L3 张量相似度并设计一个统一图卷积框架以综合多尺度信息是关键的，

* 对模型的贡献十分关键，

  

2. 实际上 L3 对模型的贡献十分关键，尽管卷积核
3. 拉普拉斯矩阵L 和 L3  









# General Response (Motivation and Contributions)

We have reorganized our motivation and distilled our key contributions as follows:

**Challenge 1: Metric collapse under HDLSS with concentration effects.**

In HDLSS settings,  pairwise distances between samples tend to concentrate, and the noise further erode discriminative power of models. As a result, conventional semi‑supervised methods struggle to learn informative representations.

- **Our solution** is to introduce new metric to describe relationships between samples.  Rather than relying solely on pairwise distances, we introduce a *tensor similarity* that directly models the angular (spatial) relationships among triples of samples. By constructing tensor similarity, Tensor‑GCN preserves the rich high-order interactions that pairwise metrics miss.

**Challenge 2: Overfitting risk due to scarce samples.**
With only a handful of labeled samples, deep models can easily overfit.

- **Our solution** is to incorporate priors to regularize learning. Specifically, we (1) sample k‑nearest neighbors to formulate pairwise similarity and (2) employ tensor similarity to encode higher‑order correlation among samples. We then employ a multi‑layer GCNs that fuses information across orders, yielding robust feature embeddings.



Further, we have sharpened our explanation around the **indecomposable high‑order similarity** introduced in Eq. (18):

$\mathcal{T}_{3_{ijk}} = 1 - \frac{\langle \mathbf{x}_i - \mathbf{x}_j,\ \mathbf{x}_k - \mathbf{x}_j \rangle}{d_{ij}\,d_{jk}} .$

- **Geometric intuition.**
  The term computes the cosine of the angle formed by the two displacement vectors $\mathbf{x}_i-\mathbf{x}_j$ and $\mathbf{x}_k-\mathbf{x}_j$, then normalizes it by the corresponding pairwise distances.  This preserves directional information that pairwise similarity alone cannot capture.
- **Why it matters.**
  In structures such as "cross‑line" patterns—where two classes lie on intersecting lines—pairwise distances cannot distinguish points lying on different lines but at similar radii.  The angular criterion in $\mathcal{T}_3$ separates them unambiguously.

We have added similarity heatmaps to the anonymous repository that compare (i) conventional pairwise similarity, (ii) decomposable tensor similarity, and (iii) our indecomposable tensor similarity.  The visual contrast highlights the richer structure captured by Eq. (18).

---

# NTdg

1. For the novelty, it just combines the RT-GCN likely function with spectral GNN theory. The response cannot fully satisfy my concern. Maybe low-sample size data can catch my eyes, but high-dimensional is a natural way to employ tensors. Could you explain more to address my concern?
2. For AUC = 1.0, I clearly know the calculation function, and I do not think there's a perfect two-label classifier that can do this. If it was my mistake, please show me your code on the anonymous site, and I would reproduce it.
3. Thank you for your new experiments comparing recent Tensor methods and explaining the computational complexity. That addressed some of my concerns.

### 回复

1. Thank you for your insightful comment. We have provided a more detailed explanation of the Tensor‑GCN motivation in "**General Response (Motivation and Contributions)** ". We hope this clarification addresses your concerns. 

2. We appreciate the reviewer’s careful check of our AUC scores and fully understand the concern. To facilitate independent verification we have added the following files to the anonymous repository

   * `y_pred.pt` – raw softmax outputs of Tensor‑GCN on the Lung dataset.
   * `test_mask.pt` - mask file indicating which portions of the dataset belong to the test set.
   * `y.pt` – ground‑truth labels (indexed 0‑4)
   * `reproduce_auc.py` – a self‑contained script that loads the three files above and prints the AUC.

    We hope these resources resolve the concern, and we remain at your disposal for any additional clarifications.

   ---

   # 2xcp

   Thank you for the author's response. Despite the author's response, I still have a few issues.

   1. Firstly, the motivation explanation provided by the authors cannot fully convince me, and as the Reviewer NTdg mentioned, its novelty needs to be considered, despite the authors's responses. The motivation behind this article may need to be re emphasized and integrated. I think maybe they can reorganize their contributions in the next version.
   2. In addition, the authors claim that sacrificing complexity leads to improved performance, which I believe may be a compromise balancing strategy, but it is also a drawback of this paper.
   3. Thirdly, thank the author for providing further experimental results. However, the experimental response provided by the authors do not include large-scale graph datasets, such as the ogbn-Arxiv dataset, which has 169,343 samples. This type of dataset can truly reflect the performance of the proposed method in real-world scenarios.

   Nevertheless, I still appreciate the authors's efforts during the rebuttal phase, including incorporating a large number of experiments.

   ### 回复

   Thank you for your feedback, hope that the below listed responses address your concerns:

   1. In the official document "**General Response (Motivation and Contributions)**" we have provided a detailed explanation of the motivation for proposing Tensor‑GCN and reorganized its key contributions. We hope this addresses your concerns.
   2. We agree that the tensor module adds extra computation.  However, in the HDLSS setting that Tensor‑GCN specifically targets, the cost remains marginal: Table 3 shows that training for 200 epochs on the largest HDLSS dataset we used (Lung, 203 samples) takes **≈ 9 s**, only **< 8 s** longer than vanilla GCN, and the gap shrinks to **< 3 s** on smaller datasets.  Thus the modest overhead is outweighed by the clear performance gains.
   3. With sub‑graph sampling, Tensor‑GCN can be trained in mini‑batches and is, in principle, able to handle much larger graphs—including datasets of the scale of ogbn‑Arxiv (169 k nodes). Extending Tensor‑GCN to large graphs is a valuable direction, we are currently validating the scalability of Tensor‑GCN on larger‑scale graph datasets.

   





---

### NTdg

1. For the novelty, it just combines the RT-GCN likely function with spectral GNN theory. The response cannot fully satisfy my concern. Maybe low-sample size data can catch my eyes, but high-dimensional is a natural way to employ tensors. Could you explain more to address my concern?
2. For AUC = 1.0, I clearly know the calculation function, and I do not think there's a perfect two-label classifier that can do this. If it was my mistake, please show me your code on the anonymous site, and I would reproduce it.
3. Thank you for your new experiments comparing recent Tensor methods and explaining the computational complexity. That addressed some of my concerns.



1. Thank you for your insightful comment. We have provided a more detailed explanation of the Tensor‑GCN motivation in "**General Response (Motivation and Contributions)** ". We hope this clarification addresses your concerns. 

2. We appreciate the reviewer’s careful check of our AUC scores and fully understand the concern. To facilitate independent verification we have added the following files to the anonymous repository

   * `y_pred.pt` – raw softmax outputs of Tensor‑GCN on the Lung dataset.
   * `test_mask.pt` - mask file indicating which portions of the dataset belong to the test set.
   * `y.pt` – ground‑truth labels (indexed 0‑4)
   * `reproduce_auc.py` – a self‑contained script that loads the three files above and prints the AUC.

    We hope these resources resolve the concern, and we remain at your disposal for any additional clarifications.



**Response to Challenge 1**



1. We fully acknowledge that tensor-base similarity has appeared in prior work, but most existing tensor‑based measures are merely pairwise similarity cast into a higher‑order format, and thus provide no extra information. Concretely, the decomposable tensor similarity (i.e. $T_3 = S\ast S$) described in eq.11 produces virtually the same affinity patterns as the underlying pairwise matrix $S$. As shown in our repository’s heatmaps ( `SYN300_dec-Tensor` versus `SYN300_pairwise`), the two are nearly indistinguishable. On the other hand, the indecomposable tensor similarity we proposed integrates the anchor‑point paradigm with angular relationships among samples, thereby capturing inter‑sample associations that go beyond simple pairwise similarity. We would be pleased to include in the manuscript a deeper theoretical analysis of the distinctions between our proposed tensor similarity and other existing tensor similarity measures.
2. There’s a slight delay in updates to the anonymous repository. We’ve just forced a refresh, and you should now be able to find the code in the `Tensor-GCN-main/reproduce_auc` folder. 



Here's a clear and respectful rebuttal that acknowledges the reviewer’s point while defending your contribution by emphasizing how your method builds upon existing ideas and what’s novel in your approach:
Thank you for the helpful observation. We fully acknowledge that angular or tensor-based similarity functions have been explored in prior work, including in the GNN literature. Our intention was not to claim the angular similarity as a wholly novel construct. Rather, our contribution lies in how we incorporate this tensor-based affinity into a unified graph learning framework to explicitly mitigate the metric collapse issue in HDLSS settings.
In particular:
Contextualization within HDLSS: While prior works use angular similarity or cosine-based metrics, we explicitly connect this formulation to the concentration effects in high-dimensional low-sample-size (HDLSS) regimes, where pairwise distances become unreliable. 



Our tensor affinity is introduced as a structural correction mechanism that captures contextual similarity (e.g., T3ijk\mathcal{T}_{3_{ijk}} models how xix_i and xkx_k align relative to anchor xjx_j), thus enhancing representation robustness under noise and sparsity.

Graph construction and propagation: Rather than using this affinity in isolation, we embed it into the graph structure to influence message passing and diffusion, allowing context-aware, high-order structure propagation. This operationalizes angular similarity in a principled and global way across the learning pipeline.
Empirical effectiveness and simplicity: Our design is computationally lightweight and integrates seamlessly into existing GCN-style architectures. The empirical results demonstrate that even this simple geometric modeling significantly improves performance, particularly under high noise and low label availability.
We will revise the manuscript to better acknowledge related work, clarify our novelty as being the integration strategy and problem setting, and draw connections to previous GNN research that inspires this design



### 2xcp

Thank you for the author's response. Despite the author's response, I still have a few issues.

1. Firstly, the motivation explanation provided by the authors cannot fully convince me, and as the Reviewer NTdg mentioned, its novelty needs to be considered, despite the authors's responses. The motivation behind this article may need to be re emphasized and integrated. I think maybe they can reorganize their contributions in the next version.
2. In addition, the authors claim that sacrificing complexity leads to improved performance, which I believe may be a compromise balancing strategy, but it is also a drawback of this paper.
3. Thirdly, thank the author for providing further experimental results. However, the experimental response provided by the authors do not include large-scale graph datasets, such as the ogbn-Arxiv dataset, which has 169,343 samples. This type of dataset can truly reflect the performance of the proposed method in real-world scenarios.

Nevertheless, I still appreciate the authors's efforts during the rebuttal phase, including incorporating a large number of experiments.



1. In the official document "**General Response (Motivation and Contributions)**" we have provided a detailed explanation of the motivation for proposing Tensor‑GCN and reorganized its key contributions. We hope this addresses your concerns.
2. We agree that the tensor module adds extra computation.  However, in the HDLSS setting that Tensor‑GCN specifically targets, the cost remains marginal: Table 3 shows that training for 200 epochs on the largest HDLSS dataset we used (Lung, 203 samples) takes **≈ 9 s**, only **< 8 s** longer than vanilla GCN, and the gap shrinks to **< 3 s** on smaller datasets.  Thus the modest overhead is outweighed by the clear performance gains.

However, in the HDLSS setting that Tensor‑GCN primarily targets, the number of samples $n$ is far smaller than the feature dimension $m$, so the method is not computationally demanding. As reported in Table 3, Tensor‑GCN requires only about **9 s** to train for 200 epochs on the largest HDLSS dataset Lung (203 samples, 3,312 features), and roughly **5 s** on the other HDLSS datasets.





1. With sub‑graph sampling, Tensor‑GCN can be trained in mini‑batches and is, in principle, able to handle much larger graphs—including datasets of the scale of ogbn‑Arxiv (169 k nodes). Extending Tensor‑GCN to large graphs is a valuable direction, we are currently validating the scalability of Tensor‑GCN on larger‑scale graph datasets.



### FB7i

Thanks for the proposed changes. First I should note that the rebuttal was limited to 2500 characters. You have (mis)used the official comment to make a longer rebuttal. This is not fair to other authors that followed the rules. I am not sure that you have addressed all notations issues. The issue is not related only to giving a reference to Von Luxburg paper.



实际上，我们重构了谱图方法的介绍，

By using the diffusion framework as you have done in the W2-2, the diffusion do not only depend on direct neighbours but on all the graph (the function g can defined over the whole graph, in this context you do not need anymore to use the L3 tensor. As said in my initial review, tensor are only useful to simplify the calculations over GPU type of architecture and do not provide new insights. I am not seeing something in the paper that will really benefit theoretically from Tensor notation. The use of tensor in the title is therefore questionable.

This reduce the interest of the paper.



**Analyzing**

Pulling in details to give the best answer

**解析图扩散与ChebNet性能差异**

用户询问为什么从图扩散的角度来说，ChebNet的表现不如GCN。GCN在图扩散过程中的表现类似低通滤波，通过一步扩散（带自环正则化的对称扩散）。而ChebNet则利用高阶的切比雪夫多项式，进行多次图扩散或高阶滤波，这会导致过拟合并影响模型的性能。因此，ChebNet由于学习到的不合法系数导致了较差的泛化能力，表现不如GCN。

**解析ChebNet与GCN的图扩散差异**

ChebNet在低阶情况下（K=2）可能学习到不合法的系数，破坏图扩散稳定性。它的重参数化方法给高频主成分赋予过大权重，容易导致过平滑或过拟合。从图扩散角度来看，GCN是基于稳定的扩散过程（例如热核逼近或随机漫步），保证了平滑性，而ChebNet的多项式近似则可能导致不稳定的扩散效应和性能下降。GCN的固定低通滤波器有效地平滑图数据。

**解析GCN与ChebNet的图扩散框架**

在图扩散框架中，GCN通过一次平滑操作（规范化邻接矩阵）实现扩散，相当于低通滤波器，能够稳定特征的平滑。相比之下，ChebNet使用多阶切比雪夫多项式逼近谱滤波器，但因为其多项式系数没有约束，可能导致高阶项的系数过大或为负值，破坏了扩散核的衰减特性，造成不稳定的震荡（Runge现象），进而导致过拟合和泛化能力差，因此在实际应用中表现不如GCN (网站)。

已思考 18 秒

在图信号处理的框架下，我们可以把 GCN 和 ChebNet 都看作是在图上做“扩散”（diffusion）或“平滑”操作，但是二者在滤波核（filter）设计和系数约束上有本质区别，导致 ChebNet 在实践中反而不如 GCN：

1. **GCN≈一次对称扩散（固定低通滤波）**
    GCN 通过加入自环并做归一化后的邻接矩阵

   P~=(D+I)−12(A+I)(D+I)−12  \widetilde P = (D+I)^{-\tfrac12}(A+I)(D+I)^{-\tfrac12}

   实现一次“漫步”或“扩散”，对应一个**固定的低通滤波器**。这种单步扩散既足够将相似节点的信息平滑融合，又不会引入高频振荡，因此在同质图上具有很好的泛化性能citeturn0file0。

2. **ChebNet≈多步扩散的多项式组合**
    ChebNet 利用 Chebyshev 多项式 $T_k(\widehat L)$ 对图卷积核进行展开：

   Y=∑k=0KTk(L^) X Wk,L^=2Lλmax⁡−I,  Y = \sum_{k=0}^K T_k(\widehat L)\,X\,W_k,  \quad \widehat L = \frac{2L}{\lambda_{\max}} - I,

   其中每一阶 $T_k(\widehat L)$ 都对应一次“$k$ 步”扩散。理论上，随着 $K$ 增大，ChebNet 能够逼近任意频域滤波器，但同时也引入了**多阶扩散的自由组合**citeturn0file0。

3. **系数非约束导致的扩散不稳定**
    实际训练中，ChebNet 对各阶多项式系数 $W_k$ 没有任何衰减或正则化约束，高阶项往往会被赋予较大的权重。这种**非法系数**不仅破坏了低通滤波的平滑本质，还会引入类似**Runge 现象**的高频振荡，使得扩散过程失去稳定性，并在训练集上过拟合、在测试集上性能下降citeturn0file0。

4. **因而 GCN 在多数数据集上表现更好**
    由于 GCN 仅保留 $k=0,1$ 的 Chebyshev 展开并对系数做严格约束，它本质上是一种稳定的低通扩散，相比之下更能避免过度振荡和过拟合，因此在真实图数据上的半监督节点分类任务中，GCN 往往优于自由度更高但不受控的 ChebNetciteturn0file0。




