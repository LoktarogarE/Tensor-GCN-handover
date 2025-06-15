# Rebuttal 初步问题整理

## Reviewer Q6oA

**Paper Strengths:**

1.The proposed method integrates high-order tensor similarity into graph convolutional networks. This approach captures complex relationships beyond pairwise connections, and thereby addressing a key limitation of traditional GCNs in HDLSS settings.

2.Extensive experiments demonstrate that Tensor-GCN consistently outperforms existing methods on multiple HDLSS datasets.

3.The paper includes ablation experiments to analyze the impact of different components, particularly the contribution of high-order tensor similarity.

**Paper Weaknesses:**

1.The proposed Tensor-GCN significantly increases computational complexity due to high-order tensor similarity calculations, which involve expensive tensor-matrix multiplications.

HDLSS 数据规模不大；实际运行时可采用并行计算优化，如使用 einsum

2.The performance of the proposed method heavily depends on the quality of the high-order similarity measure, which is computed using Euclidean distances. Given the well-known concentration effect in high-dimensional spaces, this could introduce noise and degrade the learned representations.

方法主要关注学习多个样本间的关系与空间结构；

当前也有不基于欧氏距离的方法，我们将在进一步的工作中尝试引入此类相似度。

3.The proposed Tensor-GCN integrates high-order information, which may lead to overfitting, especially in HDLSS settings where the number of samples is extremely limited.

正如您在Q2 中所说，高维空间的集中效应将会增加 overfitting 的风险，这是我们引入张量相似度的动机之一，因为张量相似度能为模型提供更多有效信息。

4. The authors are suggested to include experiments on benchmark graph datasets and provide further insights into computational efficiency.

#### Reviewer Q6oA

1. 由于涉及昂贵的 tensor-matrix 乘法计算，所提出的 Tensor-GCN 显著增加了计算复杂度。
2. **所提出方法的性能在很大程度上依赖于高阶相似性度量的质量，该度量是使用 Euclidean distances 计算的。考虑到高维空间中众所周知的集中效应，这可能会引入噪声并降低所学习表示的质量。**
3. 所提出的 Tensor-GCN 整合了高阶信息，这可能导致过拟合，尤其是在样本数量极其有限的 HDLSS 设置中。
4. 建议作者在基准图数据集上包含实验，并提供更多关于计算效率的深入见解。







#### Reviewer Lr1z

**Paper Strengths:**

- The paper is well-written.
- The clarification of notations and the problem is clear and meaningful for real-world applications.
- The experiments are well-designed, demonstrating promising performance against various baselines.

**Paper Weaknesses:**

- The related work section is missing, resulting in a limited discussion of recent relevant research.
- The implementation details and code are missing, making it challenging to reproduce the reported results based on the current description.
- It is highly recommended to include an algorithm table to illustrate the overall graph convolution process.

**Questions And Suggestions For Rebuttal:**

- The selected baselines in the comparison experiments should include the most recent state-of-the-art methods published in 2024. The current baselines are insufficient to accurately demonstrate the effectiveness of the proposed method.
- The motivation behind this paper remains unclear due to the brief discussion of challenges in the introduction. Providing an example to support your statement would help illustrate the contributions of this paper more intuitively.

#### Reviewer Lr1z

1. 缺少相关工作部分，导致对近期相关研究的讨论较为有限。
2. 缺少实现细节和代码，基于现有描述难以复现报告的结果。
3. **强烈建议添加算法表以展示整体的 graph convolution 过程。**
4. 对比实验中选择的基线方法应包括 2024 年发表的最新 state-of-the-art 方法。目前的基线不足以准确展示所提方法的有效性。
5. 由于引言中对挑战的讨论较为简略，论文的动机仍不清晰。提供一个示例来支持您的论述将有助于更直观地展示论文的贡献。



#### Reviewer FB7i

**Paper Strengths:**

The use of kNN graph in the analysis of high dimensional data is a very popular approach for reducing a complex multidimensional space into a tractable metric graph. However, this approach while easy to implement, have a major drawback; it assume somewhat arbitrarily that interactions are only point to point, and the scope of these interactions is only hard limited to k closest neighbours. Therefore any approach that helps in integrating higher order relations is reducing the negative impacts of the kNN will be valuable. The proposed graph convolution structure is promising. The ablation analysis in section 4.3 clearly shows its interest. The performance improvement presented also validate the interest of the method.

**Paper Weaknesses:**

As always in Tensor based paper, the main weakness is notation and assumption about reader knowledge. For example in section 2, in begins by noting "input data \bf X", and talks about the Laplacian. But without stating that there is a kNN graph construction there, the reader assumes that the Laplacian is defined over the full pairwise distance matrix, and if there is a graph built, it should be a complete graph. The kNN is not introduced (and even never defined) before section 2.3.

-The use of Fourier terms here is also misleading. Indeed Fourier transform is a specific example of linear space decomposition named "Spectral methods", but we are not using here a Fourier transform, with the frequency interpretations. A better reference to spectral methods in graph is needed. For example the well-known "Ulrike Luxburg. 2007. A tutorial on spectral clustering. Statistics and Computing 17, 4 (December 2007), 395–416. " -Overall the graph convolution is a specific type of diffusion process over graph, and the technique in the paper, and the higher orders introduction can be reformulated, with simpler notations, and more intuition from a graph theory perspective, as a diffusion process. The only added value of the tensor is easing implementation of parallel processor with tensor based SIMD architectures like GPUs. The relation to diffusion should be state in the paper.

- The notation used in Eq. (16) is never defined, nor common. So it should be defined
- While this is not the role of the paper to describe competitor approaches used in the evaluation, however a minimal description of how they work and what differentiate them with the proposed methods is mandatory. For example GRACES technique that is frequently the most competitive contender of the paper's method is only described in section 4.1.2 as " a method specifically tailored for HDLSS data, using GCN to iteratively find a set of optimal features.". This clearly not enough to make the paper self contained.

#### Reviewer 2xcp

**Paper Weaknesses:**

Although this job is quite interesting, I think it still has some weaknesses:

1. In the abstract and introduction, although the basic background of GCN is introduced, the motivation of high-order tensors has not been fully discussed. In addition, there has been a significant amount of work done to address overfitting issues in GCN. If the motivation for this work cannot be well reconstructed and discussed, then its contribution will be incremental.
2. This work lacks a lot of comparison and discussion of related work. It is suggested to add a related work Section.
3. The algorithm complexity of this method seems to have reached the third power level (due to tensor stacking), and I am concerned about its adaptability in large-scale real-world scenarios.
4. The dataset used in the experiment is also relatively small in scale and lacks experimental comparisons on real graph datasets.
5. This work lacks more state-of-the-art comparative methods to demonstrate the effectiveness of the proposed method.
6. I think there is still more room for improvement in this work, including writing, grammar, experimental setup, and so on.

In short, I believe this work still has room for improvement before it meets the acceptance criteria for KDD.

#### Reviewer 2xcp

**论文缺点:**

虽然这项工作非常有趣，但我认为它仍然存在一些缺点：

1. 在摘要和引言中，虽然介绍了 GCN 的基本背景，但对高阶 tensor 的动机讨论不够充分。此外，已有大量工作致力于解决 GCN 中的过拟合问题。如果无法充分重构和讨论这项工作的动机，其贡献将仅是渐进式的。
2. 该工作缺乏大量相关工作比较和讨论，建议增加一个相关工作部分。
3. 该方法的算法复杂度似乎达到了三次方级别（由于 tensor stacking），我担心其在大规模实际场景中的适应性。
4. 实验中使用的数据集规模相对较小，缺乏在真实图数据集上的实验对比。
5. 该工作缺乏更多 state-of-the-art 的比较方法来展示所提方法的有效性。
6. 我认为这项工作在写作、语法、实验设置等方面仍有改进空间。

总之，我认为在达到 KDD 接受标准之前，这项工作仍有提升空间。

---

#### Reviewer FB7i

**论文优点:**

在高维数据分析中使用 kNN graph 是一种非常流行的方法，用于将复杂的多维空间简化为可处理的度量图。然而，这种方法虽然易于实现，却有一个主要缺点；它在某种程度上武断地假设交互仅限于点对点，并且这些交互的范围仅严格局限于 k 个最近邻。因此，任何有助于整合高阶关系以减少 kNN 负面影响的方法都将是有价值的。所提出的 graph convolution 结构很有前景。第 4.3 节中的消融分析清晰地展示了其优势。所展示的性能提升也验证了该方法的价值。

**论文缺点:**

正如基于 Tensor 的论文一贯存在的，主要缺点在于符号的使用和对读者知识的假设。例如，在第 2 节中，文中以 "input data **X**" 开始，并讨论了 Laplacian。但未说明此处存在 kNN graph 构造，读者会认为 Laplacian 是基于完整的成对距离矩阵定义的，而如果构建了图，则应为完全图。kNN 在第 2.3 节之前未被介绍（甚至从未定义）。

- 这里使用 Fourier 术语也容易产生误导。实际上，Fourier transform 是一种名为 "Sp	ectral methods" 的线性空间分解的特定例子，但我们在这里并未使用具有频率解释的 Fourier transform。需要对图中的 spectral methods 提供更好的参考。例如著名的 "Ulrike Luxburg. 2007. A tutorial on spectral clustering. Statistics and Computing 17, 4 (December 2007), 395–416."
  ——总体而言，graph convolution 是图上的一种特定类型的扩散过程，本文中的技术以及高阶信息的引入可以重新表述为扩散过程，采用更简单的符号和更多来自图论角度的直观解释。Tensor 唯一的附加价值在于便于在基于 Tensor 的 SIMD 架构（如 GPU）上实现并行处理。文中应明确说明其与扩散过程的关系。
- Eq. (16) 中使用的符号从未定义，也不常见，因此应予以定义。
- 虽然描述竞争方法不是论文的主要任务，但必须对它们的工作原理以及与所提方法的差异进行最基本的描述。例如，GRACES 技术作为论文方法中竞争性最强的 contender，在第 4.1.2 节中仅描述为 "a method specifically tailored for HDLSS data, using GCN to iteratively find a set of optimal features." 这显然不足以使论文自包含。





#### Reviewer NTdg

**Paper Weaknesses:**

1. Although the idea presents some interesting aspects, the novelty of using a third-order tensor to capture the high-order relationships is somewhat limited. In recent years, numerous tensor-based graph convolutional networks have been proposed. In my opinion, the most significant innovation of employing a tensor polynomial filter is not entirely new. Additionally, this paper merely utilizes a second-order ChebyshevNet with a folding tensor approach, and its corresponding multiplication operation demonstrates limited innovation.
2. The selected baselines are not sufficiently new, where the most recent approach is publicated in the year 2023. Moreover, the authors failed to compare Tensor-GCN with other tensor-based GCN methods, such as MR-GCN [1], RT-GCN [2], TM-GCN [3], etc. Additionally, they did not cite these tensor-based GCNs in their paper.
3. The paper does not discuss the computational complexity of tensor operations, and the proposed method seems difficult to scale to large datasets.
4. In Table 2, the AUC of Tensor-GCN on the Lung dataset is 1.000. I strongly doubt this result and request a reproduction of this dataset.

   这类结果确实不常出现，但注意到 Lung 数据集有 5 种不同类别，对于非二分类任务，我们采用 OvR（One-vs-Rest）策略来计算AUC，这意味着只要其中部分类被模型完全区分，AUC 的值确实能达到 1，这是因为模型对阳性和阴性样本在预测分数上实现了“无重叠分布”，即所有阳性分数都比阴性分数高（或相反）

[1] Huang, Zhichao, et al. "MR-GCN: Multi-Relational Graph Convolutional Networks based on Generalized Tensor Product." IJCAI. Vol. 20. 2020. [2] Wu, Zhebin, et al. "Robust tensor graph convolutional networks via t-svd based graph augmentation." Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022. [3] Malik, Osman Asif, et al. "Dynamic graph convolutional networks using the tensor M-product." Proceedings of the 2021 SIAM international conference on data mining (SDM). Society for Industrial and Applied Mathematics, 2021.

------

#### Reviewer NTdg

**论文缺点:**

1. 尽管该思路呈现了一些有趣的方面，但使用三阶 tensor 来捕捉高阶关系的创新性有限。近年来，已经提出了众多基于 tensor 的 graph convolutional networks。在我看来，采用 tensor polynomial filter 的最显著创新并不完全新颖。此外，本文仅利用了第二阶 ChebyshevNet 结合 folding tensor 方法，其对应的乘法操作显示出创新性有限。
2. 所选基线方法不够新颖，其中最新的方法发表于 2023 年。此外，作者未能将 Tensor-GCN 与其他基于 tensor 的 GCN 方法进行比较，如 MR-GCN [1]、RT-GCN [2]、TM-GCN [3] 等。同时，他们也未在论文中引用这些基于 tensor 的 GCN。
3. 论文未讨论 tensor 操作的计算复杂度，所提方法似乎难以扩展到大规模数据集。
4. 在表 2 中，Tensor-GCN 在 Lung 数据集上的 AUC 为 1.000。我对这一结果深表怀疑，并要求重新实验该数据集。

[1] Huang, Zhichao, et al. "MR-GCN: Multi-Relational Graph Convolutional Networks based on Generalized Tensor Product." IJCAI. Vol. 20. 2020. [2] Wu, Zhebin, et al. "Robust tensor graph convolutional networks via t-svd based graph augmentation." Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 2022. [3] Malik, Osman Asif, et al. "Dynamic graph convolutional networks using the tensor M-product." Proceedings of the 2021 SIAM international conference on data mining (SDM). Society for Industrial and Applied Mathematics, 2021.

>  这篇TM-GCN去年有人问过，这是当时的回应：
>
> "Dynamic Graph Convolutional Networks Using the Tensor M-Product": This paper focuses on dynamic graphs over time, whereas Tensor-GCN aims to measure and leverage the intrinsic connections of static data to achieve accurate semi-supervised classification in HDLSS environments.
