# Reviewer Q6oA

> **W1 & Q1: Increased computational complexity.**

Thank you for your comment. While our high-order tensor similarity does increase computational cost compared to pairwise methods, our experiments (see Table 3) show that the runtime increase is modest relative to the significant gains in performance for HDLSS data. We are also exploring optimization strategies such as sparse representations and parallelization. Please refer to our official comment "**General Response (Complexity and Scalability Analysis)**" for further details.

> **W2 & Q2: Potential noise from Euclidean-based similarity.**

We appreciate the reviewer’s insight regarding the reliance on Euclidean distances when constructing the high-order similarity measure. The primary motivation behind Tensor-GCN is to model the spatial relationships among multiple samples, thereby alleviating the adverse effects of the concentration phenomenon in high-dimensional spaces.

Additionally, we have visualized the embedding matrices for various similarity measures and observed improvements in discriminative performance. These results, documented in our anonymous repository, indicate that the tensor similarity employed in our framework does not introduce significant noise and indeed facilitates robust feature extraction.

Due to the length restrictions of the rebuttal, further explanation will be provided via the **official comment**.

> **W3 & Q3: Risk of overfitting in HDLSS settings.**

We appreciate the reviewer’s concern regarding potential overfitting. It is true that all methods are challenged by overfitting in HDLSS settings. However, a key strength of Tensor-GCN is its use of tensor similarity to model the intrinsic spatial relationships among multiple samples. This additional high-order information serves as complementary information to traditional pairwise measures, thus helping to alleviate the overfitting problem by offering a richer representation of the sample relationships. 

> **W4 & Q4: Need for experiments on graph datasets and efficiency analysis.**

We appreciate your suggestion. In our offical comment "**General Response (Supplementary Experiments)**", we provide comparative experiment on benchmark graph dataset Cora to further illustrate our method's performance. 

To address your concerns regarding computational efficiency, we recommend referring to our offical comment "**General Response (Complexity and Scalability Analysis)** "





# Reviewer Lr1z

We gratefully thank you for your valuable comments! Below we give point-by-point responses to your comments. Hope that our responses could address your concerns!

> **W1: Missing related work discussion.**

(prev)We appreciate your observation. Due to strict page limitations, we provided only a brief overview of related research in the introduction. Tensor-GCN belongs to the family of spectral graph convolutional networks. Specifically, it endeavors to uncover high-order relationships between samples and aids in modeling the discrimination of high-dimensional data. We recognize the importance of a more comprehensive discussion and plan to expand on this context in future work or supplementary materials.



We recognize the importance of related work section. Due to strict page limitations, we were unable to include a dedicated related work section in the initial submission. However, we have revised the manuscript to allocate sufficient space for this section, and we will incorporate a more extensive discussion of related work in the updated version.

我们认可相关工作章节的重要性，但由于篇幅限制，我们未能在最初的版本中提供专门的 相关工作章节。我们已经修改了手稿，为相关工作章节腾出空间，我们会在修改版的手稿中加入更多相关工作的讨论。

> **W2: Absent implementation details and code.**

We acknowledge the importance of reproducibility. In our manuscript, we include anonymous links to both the code and datasets, along with all necessary parameter details for replication. We are currently refactoring the codebase of Tensor-GCN and it will be made publicly available promptly upon paper acceptance.

> **W3: Lacks algorithm table.**

We appreciate the suggestion, due to the length limit for the rebuttal, we will provide the model workflow later via official comment.



The overall workflow of Tensor-GCN:

**Input**: Node feature matrix ${X}$; Labeled node set $Y_L$  
**Parameters**: Number of layers $N$; Number of epochs $T$; Learning rate $\eta$  
**Output**: Predicted labels for all nodes $\hat{Y}_{pred}$  

1. Compute the normalized second-order Laplacian matrix $\hat{{L}}$
2. Compute the folded normalized third-order Laplacian tensor ${L}_3$ 
3. **for** $t = 0,1,\dots, T-1$ **do**
   1. **for** $l = 0, 1, \dots, N-1$ **do**
      1. Compute second-order feature: ${H}_1^{(l)} \gets \hat{{L}}{X}^{(l)}$
      2. Compute third-order feature: ${H}_2^{(l)} \gets \Bigl({L}_{3}^T\bigl({X}^{(l)} * {X}^{(l)}\bigr)\Bigr) - {X}^{(l)}$
      3. Concatenate features to update layer output: $Z^{(l)} \gets \sigma([{X}^{(l)},\,{H}_1^{(l)},\,{H}_2^{(l)}]W^{(l)})$
      4. Compute cross-entropy loss on $Y_L$: $\text{Loss} = -\sum_{i\in Y_L}\sum_{j=1}^{F} Y_{ij}\ln Z_{ij}$
      5. Update all layer weights ${W}^{(l)}$ via backpropagation using learning rate $\eta$
   2. **end for**
4. **end for**
5. For each node $i$, $\hat{y}_i \gets \arg\max_j Z_{ij}$
6. **return** Predicted labels $\hat{Y}_{pred}$

> **Q1: Need for SOTA baselines published in 2024.**

We appreciate the reviewer’s suggestion. Please refer to our offical comment "**General Response (Supplementary Experiments)**".

> **Q2: Unclear motivation with insufficient examples.**

We recognize the need for a clear motivation. In fact, the challenge in high-dimensional spaces is that the concentration effect causes a collapse of distances between samples, rendering pairwise similarity measures less effective at distinguishing different samples. To overcome this, we introduce tensor similarity to capture relationships among multiple samples. In our anonymous repository, we have included updated heatmap visualizations comparing different order similarities. These results demonstrate that incorporating indecomposable tensor similarity significantly enhances the model's discriminative power, thereby providing a more intuitive illustration of our contributions.



# Reviewer FB7i

Thank you for your thoughtful feedback! We hope the following response will be helpful to you:

> **W1: Assumption about reader knowledge;  Talk about the Laplacian before introducing kNN graph construction.**

We apologize for the confusion regarding the graph construction. We agree that it is essential to clarify that our method begins by constructing a kNN graph for the input data before discussing the Laplacian and related operations. In the revised manuscript, we have updated Section 2 to explicitly introduce the kNN-based graph construction, ensuring that readers understand that the Laplacian is derived from this kNN graph rather than a full pairwise distance matrix. 

> **W2: Misleading use of Fourier terms without proper spectral reference; the relation to diffusion should be state in the paper.**

We thank the reviewer for these constructive suggestions. In the revised manuscript, we have restructured the introduction to spectral graph convolution by drawing on the reference suggested (Luxburg, 2007)

Due to the length constraints of the rebuttal, the connection between Tensor-GCN and diffusion will be addressed in the **official comment**.

> **W3: Undefined notation in Eq. (16).**

We appreciate the reviewer’s comment. In Eq. (16), the operator “||” denotes the concatenation of feature representations from three distinct domains: (i) the self-information $Y_0(x)$; (ii) the first-order neighborhood information $Y_1(x)$ derived from the conventional Laplacian; and (iii) the high-order neighborhood information $Y_2(x)$ obtained via the tensor similarity-based Laplacian tensor. The concatenated feature vector is then multiplied by the learnable parameter $\Theta$, which integrates these multiple sources of information into the final output. We will revise the manuscript to explicitly define this notation for clarity.

> **W4：Insufficient description of competitor methods.**

We thank the reviewer for pointing out the need for additional minimal descriptions of competitor approaches. We have revised the manuscript to provide a more detailed introduction of the baseline methods. For example, GRACES leverages graph convolutional networks to build dynamic similarity graphs, enabling it to iteratively select the most informative features from HDLSS data. It mitigates overfitting by integrating multiple dropout strategies, Gaussian noise injection, and ANOVA F-test-based correction into its gradient-driven feature selection process.



# Reviewer 2xcp

Thank you for your critical comments. Your constructive criticism has been invaluable in refining our work. Below, we give point-by-point responses to your comments:

> **W1: The motivation of high-order tensors has not been fully discussed**

We appreciate the reviewer’s insightful comments. In high-dimensional settings, the well-known concentration effect causes distances between samples to collapse, making pairwise similarity measures insufficient for distinguishing between different samples. Although methods such as hypergraph-based approaches have been proposed to mitigate overfitting in GCNs, they fundamentally rely on pairwise similarities and thus do not fully overcome this limitation. To address this, we introduce a tensor similarity measure that captures the relationships among multiple samples simultaneously. 

We provide heatmap visualizations of different similarities in the anonymous repository, which demonstrate that using a third-order indecomposable similarity markedly improves the model’s discriminative capabilities. We believe that these enhancements substantively strengthen the motivation and contribution of our work.

> **W2: Lacks a dedicated related work section.**

(prev)We thank the reviewer for this insightful suggestion. In our current version, we have integrated a brief discussion of high-order extensions to GCNs within the introduction, outlining methods that inspire the development of Tensor-GCN. Given the strict page constraints of the current submission, allocating a dedicated related work section would require omitting other vital content. We believe the concise discussion in the introduction sufficiently contextualizes our contributions within the existing literature. In future revisions or extended versions, we will consider expanding this discussion to provide a more comprehensive comparison with related work.



We appreciate your suggestion. We recognize the importance of a dedicated related work section, however, due to strict page limitations, we were unable to include one in the initial submission. We have revised the manuscript to allocate sufficient space for this section, and we will incorporate a more extensive discussion of related work in the updated version.

> **W3 : High algorithm complexity, raising scalability concerns.  **

We appreciate the reviewer’s concern regarding computational complexity. The reviewer may refer to our  official comment "**General Response (Complexity and Scalability Analysis)**" for further details.

> **W4 & W5 : missing real graph comparisons; missing state-of-the-art comparative methods.**

Please refer to our  official comment "**General Response (Supplementary Experiments)**".

> **W6: Overall need for improvements in writing and experimental setup.**

We appreciate your valuable feedback and understand that there is still room for improvement in several aspects of the manuscript. We will carefully review the text to enhance clarity and language quality and will consider your suggestions to further refine the experimental design and presentation. Your constructive comments will definitely be taken into account in our revised version.



#### Reviewer NTdg

We gratefully thank you for your valuable comments! Below we give point-by-point responses to your comments. Hope that our responses could address your concerns!

> **W1: Limited novelty in high-order tensor usage.** 「这个估计得再改下」

We appreciate your perspective regarding the novelty of using a third-order tensor. While prior work has explored tensor-based GCNs, our key contribution lies in integrating the indecomposable third-order similarity within a polynomial filtering framework specifically tailored for HDLSS data. This approach enriches the representational capacity beyond pairwise methods, and our folding‐based convolution architecture offers a more direct utilization of higher‐order structure. Experimental results further validate that this synergy effectively enhances model robustness and discrimination under the challenging HDLSS setting.

> **W2: Outdated baselines; missing tensor-based GCN comparisons.**

We thank the reviewer for the constructive suggestions. We have updated our experimental evaluation to include the latest methods published in 2024 as well as RT-GCN as additional baselines. The reviewer may refer to our official comment "**General Response (Supplementary Experiments)**" for further details. 

Furthermore, we have noted the emergence of several new tensor-based GCN approaches and have expanded the discussion in the manuscript to compare and contrast these methods with Tensor-GCN.

> **W3: No complexity discussion; scalability issues unaddressed.**

Please refer to our offical comment "**General Response (Complexity and Scalability Analysis) **".

> **W4: Questionable AUC result on Lung dataset**

We appreciate the reviewer's close attention to our results. For the multi-class Lung dataset—which contains five distinct classes—we compute the AUC using the One-vs-Rest (OvR) strategy. Under this evaluation, the AUC for each binary sub-problem is calculated separately. In cases where the model achieves a complete separation between the positive samples of a given class and the negatives (i.e., the predicted scores for positive samples are entirely higher than those for negative samples), the corresponding AUC will be 1. Consequently, if one or more of these binary classifications perfectly differentiate the classes, the overall reported AUC can indeed reach 1.

We have verified these findings through reproductions of the experiments. To address potential concerns, we have expanded the manuscript to include further details regarding the OvR methodology.

---

# 官方评论

# General Response (Supplementary Experiments)  

*Dear Reviewers,*  

 We have noted that several reviewers commented on the lack of comparisons with the latest baselines and tensor-based GCN methods. In response, we have introduced RT-GCN [1] (2022) and CHGNN [2] (2024) as additional baseline methods. Below, we present the experimental results that compare our approach with these methods: 

| Dataset         | Method           | ACC (%)    | F-Score    | AUC        | Recall     |
| :-------------- | :--------------- | :--------- | :--------- | :--------- | :--------- |
| **Leukemia**    | RT-GCN           | 75.864     | 0.7181     | 0.6958     | 0.7181     |
| **Leukemia**    | CHGNN            | 86.211     | 0.7333     | 0.8042     | 0.6111     |
| **Leukemia**    | Tensor-GCN(Ours) | **89.655** | **0.9927** | **0.9444** | **0.9280** |
| **ALLAML**      | RT-GCN           | 72.413     | 0.6869     | 0.7014     | 0.6931     |
| **ALLAML**      | CHGNN            | 82.765     | 0.6429     | 0.7583     | 0.5000     |
| **ALLAML**      | Tensor-GCN(Ours) | **83.908** | **0.8824** | **0.8610** | **0.8869** |
| **GLI_85**      | RT-GCN           | 82.351     | 0.7809     | 0.8500     | 0.7729     |
| **GLI_85**      | CHGNN            | 82.351     | 0.8605     | **0.8677** | 0.7708     |
| **GLI_85**      | Tensor-GCN(Ours) | **84.559** | **0.8718** | 0.8250     | **0.8765** |
| **Prostate_GE** | RT-GCN           | 75.610     | 0.7524     | 0.7620     | 0.7561     |
| **Prostate_GE** | CHGNN            | 76.834     | 0.7816     | 0.8042     | 0.8095     |
| **Prostate_GE** | Tensor-GCN(Ours) | **83.961** | **0.9412** | **0.9411** | **0.9412** |
| **Lung**        | RT-GCN           | 75.463     | 0.3350     | 0.8707     | 0.3760     |
| **Lung**        | CHGNN            | 93.250     | 0.7523     | 0.9885     | 0.7767     |
| **Lung**        | Tensor-GCN(Ours) | **94.070** | **0.9864** | **1.000**  | **0.9875** |

We also evaluated the performance of Tensor-GCN on regular data. Specifically, we sampled 400 instances from the graph dataset Cora and the image dataset USPS, respectively. The table below compares the accuracy of Tensor-GCN with that of standard GCN.

| Dataset  | Method     | ACC (%) |
| :------- | :--------- | :------ |
| **Cora** | GCN        | 79.573  |
| **Cora** | Tensor-GCN | 82.690  |
| **USPS** | GCN        | 86.083  |
| **USPS** | Tensor-GCN | 87.250  |

The experimental results indicate that Tensor-GCN adapts well to conventional data, and its performance under HDLSS settings surpasses that of the latest proposed GCN methods.

[1] Wu, Zhebin, et al. "Robust tensor graph convolutional networks via t-svd based graph augmentation." *Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining*. 2022.

[2] Song, Yumeng, et al. "CHGNN: a semi-supervised contrastive hypergraph learning network." *IEEE Transactions on Knowledge and Data Engineering* (2024).

---

*We extend our sincere gratitude to all reviewers for your invaluable time and dedication. Your insightful feedback has significantly enriched the refinement of Tensor-GCN, and we are profoundly grateful for your contributions.*

*Best,*

*Authors*

# General Response (Complexity and Scalability Analysis)

The complexity analysis of our method is conducted and included in the script. There are two main modules in Tensor-GCN that significantly consume time and space resources: the graph construction step and the layer-wise Tensor-GCN model. 

Specifically, given input feature $\textbf X \in \mathbb R^{N\times C}$, the time complexity of constructing a pair-wise $k$NN graph and computing the Laplacian matrix is $O(N^2)$, while obtaining the Laplacian tensor from $\mathbf X$ has a computational complexity of $O(N^3)$. Then for an $l$-layer Tensor-GCN model, its inter-layer iteration method is given by Eq. (21). The time consumption for a single layer is $O(N^3C)$, hence the computational cost for an $l$-layer Tensor-GCN model amounts to $O(lN^3C)$. Despite Tensor-GCN does exhibit higher computational complexity, the added cost yields accurate classification performance and stable results.

We appreciate the reviewer’s concerns regarding the scalability of Tensor-GCN. To address these issues, we have incorporated several techniques to enhance the efficiency of Tensor-GCN for larger datasets:

* **Batch Processing and Sparse Storage:** These methods help to handle large amounts of data more efficiently.
* **GPU Parallel Computing:** By leveraging the power of GPUs, we accelerate the computation process.

* **Subgraph Sampling for Large-Scale Graph Data:** For datasets (e.g., Cora), we employ subgraph sampling techniques during training.

Although Tensor-GCN is primarily designed for HDLSS data, these enhancements enable its application in large-scale scenarios. In the official comment "**General Response (Supplementary Experiments)**", we report results on both the image dataset (USPS) and the graph dataset (Cora), demonstrating that Tensor-GCN achieves competitive performance on standard datasets. 

---

# **W2 & Q2: Potential noise from Euclidean-based similarity.**

*Due to the length constraints of the rebuttal, we will continue our discussion on the similarity measure employed in Tensor-GCN in the official comment.*

We thank the reviewer for raising this important point. While it is true that raw Euclidean distances are prone to the concentration effect in high-dimensional spaces, our proposed high-order similarity measure in Eq. (18) effectively mitigates this issue through a normalized geometric formulation:

$\mathcal{T}_{3_{ijk}} = 1 - \frac{\langle x_i - x_j,\ x_k - x_j \rangle}{d_{ij} \, d_{jk}}$

This formulation computes the cosine of the angle between the relative displacement vectors $x_i - x_j$ and $x_k - x_j$, normalized by the pairwise distances $d_{ij}$ and $d_{jk}$. 

This approach captures data structures that conventional pairwise similarity measures fail to describe. **For example, in datasets exhibiting a cross-line structure, evaluating the angle among three points effectively differentiates distinct classes distributed along two separate lines.**

By focusing on the angular relationships and relational geometry among triplets, our method reduces reliance on the absolute Euclidean magnitudes and alleviates the adverse effects of concentration. 

---

# **W2-2: The relation to diffusion should be state in the paper.**

*Due to the length restrictions of the rebuttal, further discussion on the **diffusion** concept in Tensor-GCN is provided via official comment.*

From a mathematical standpoint, graph convolution are often interpreted as executing a diffusion process over a graph. In traditional spectral graph convolution theory, given a feature $x \in \mathbb{R}^N$, a single graph diffusion or message passing step can be expressed as:

$\mathcal{O}(L)(x) = g(L)x,$

where $g(\cdot)$ is a filter function (e.g., a polynomial function) applied to the Laplacian $L$. Viewed through the lens of diffusion, this operation can be considered as performing one or more steps of propagation, smoothing, and aggregation of the feature $x$. Essentially, conventional GCN approximates $g(L)$ (typically by a first-order or second-order approximation) to achieve diffusion among the nodes.

Our proposed method extends this concept by incorporating a third-order Laplacian tensor $\mathcal{L}_3$ that captures the joint relationships among multiple samples. This allows us to incorporate high-order neighborhood information into the diffusion framework. Specifically, our approach can be interpreted as performing a multi-order diffusion process, which can be written as:

$g(\mathcal{L}, \mathcal{L}_3) = g_1(\mathcal{L})x + g_2(\mathcal{L}_3)x.$

Here, $g_1(\mathcal{L})x$ represents the diffusion process over the traditional Laplacian matrix, while $g_2(\mathcal{L}_3)x$ captures additional high-order diffusion paths enabled by the Laplacian tensor. Through this combined process, Tensor-GCN not only benefits from the classical random walk or low-pass filtering interpretation inherent in standard GCNs, but also leverages the enhanced feature propagation provided by the third-order tensor. This improved multi-order fusion contributes to stronger feature expression and better generalization, which is particularly beneficial under HDLSS setting.

---

> **W3: Lacks algorithm table.**

The overall workflow of Tensor-GCN:

**Input**: Node feature matrix ${X}$; Labeled node set $Y_L$  
**Parameters**: Number of layers $N$; Number of epochs $T$; Learning rate $\eta$  
**Output**: Predicted labels for all nodes $\hat{Y}_{pred}$  

1. Compute the normalized second-order Laplacian matrix $\hat{{L}}$
2. Compute the folded normalized third-order Laplacian tensor ${L}_3$ 
3. **for** $t = 0,1,\dots, T-1$ **do**
   1. **for** $l = 0, 1, \dots, N-1$ **do**
      1. Compute second-order feature: ${H}_1^{(l)} \gets \hat{{L}}{X}^{(l)}$
      2. Compute third-order feature: ${H}_2^{(l)} \gets \Bigl({L}_{3}^T\bigl({X}^{(l)} * {X}^{(l)}\bigr)\Bigr) - {X}^{(l)}$
      3. Concatenate features to update layer output: $Z^{(l)} \gets \sigma([{X}^{(l)},\,{H}_1^{(l)},\,{H}_2^{(l)}]W^{(l)})$
      4. Compute cross-entropy loss on $Y_L$: $\text{Loss} = -\sum_{i\in Y_L}\sum_{j=1}^{F} Y_{ij}\ln Z_{ij}$
      5. Update all layer weights ${W}^{(l)}$ via backpropagation using learning rate $\eta$
   2. **end for**
4. **end for**
5. For each node $i$, $\hat{y}_i \gets \arg\max_j Z_{ij}$
6. **return** Predicted labels $\hat{Y}_{pred}$

---



