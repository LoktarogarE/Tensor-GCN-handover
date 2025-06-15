|            | Acceptance        | Relevant | Innovative      | Technical quality | Presentation  | of interest | Confidence | is Top 10% |      |      |
| ---------- | ----------------- | -------- | --------------- | ----------------- | ------------- | ----------- | ---------- | ---------- | ---- | ---- |
| Reviewer 1 | 3: should accept  | Yes      | 3 (Innovative)  | 3 (High)          | 3 (Good)      | 3 (Yes)     | 0 (Low)    | No         |      |      |
| Reviewer 6 | 3: should accept  | Yes      | 3 (Innovative)  | 3 (High)          | -2 (Marginal) | 3 (Yes)     | 2 (High)   | Yes        |      |      |
| Reviewer 2 | -2: marginal      | Yes      | -2 (Marginally) | 3 (High)          | 3 (Good)      | 3 (Yes)     | 0 (Low)    | No         |      |      |
| Reviewer 4 | -2: marginal      | Yes      | -2 (Marginally) | 3 (High)          | 3 (Good)      | 3 (Yes)     | 1 (Medium) | No         |      |      |
| Reviewer 3 | -4: should reject | Yes      | -2 (Marginally) | -2 (Marginal)     | 3 (Good)      | 3 (Yes)     | 2 (High)   | No         |      |      |
| Reviewer 5 | -4: should reject | Yes      | -2 (Marginally) | -4 (Low)          | -2 (Marginal) | 2 (May be)  | 1 (Medium) | No         |      |      |

```
We regret to let you know that your submission with paper ID DM732, entitled "Tensor Graph Convolutional Networks for High-dimensional and Low-sample Size Data", has not been accepted to the IEEE International Conference on Data Mining (ICDM) 2024. From all 604 complete submissions, the program committee finally accepted 66 regular papers and 52 short papers, resulting in a final acceptance rate of 19.5%. Many of the decisions were extremely difficult, the program committee endeavored to arrive at the best possible decisions. 

Due to the large number of high-quality submissions and the very limited number of papers that can be presented at the conference, we were forced to reject many interesting and potentially impactful papers. The program committee strived to ensure a fair and thorough decision process. Each paper was evaluated by at least three independent reviewers, and was discussed among the reviewers and the Area Chair. For difficult cases, additional Area Chairs and/or the Program Chairs were consulted. 

You can find the final reviews for your paper at the bottom of this message (some submissions violated the submission policy and were desk-rejected without review comments).  We sincerely hope that the reviews will help improve your paper for a subsequent strong resubmission. Despite this outcome, we hope to see you at ICDM 2024, which will be held in person in Abu Dhabi, UAE, from December 9 to 12, 2024. We would also like to draw your attention to the workshops at ICDM 2024 (https://icdm2024.org/workshops/), which focus on timely and exciting topics in data mining.  Please note that the workshop paper submission deadline is September 10, 2024.  Conference registration is already open at https://icdm2024.org/registration/ .

With best wishes,
Elena Baralis & Kun Zhang
ICDM 2024 Program Chairs

=====================================================================

        --========  Review Reports  ========--

The review report from reviewer #1:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [X] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [X] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High)
  [_] 1 (Medium)
  [X] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [X] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  Abs: TGCN has the ability to describe relationships between samples.

TGCN Looks beyond pairwise relationships to higher order: for instance it can
use a 3rd order tensor to capture the similarities between three samples.

I found the math to get in the way rather than help: needs to be significantly
reduced so as to concentrate on the key parts that help accuracy.

Results show small but consistent improvement over previous GCNs.

Would have liked internal ablation experiments comparing k=0 v k=1 v k=2
accuracies for TGCN so as to justify moving to 3rd order interactions. You can make
room by shortening the Preliminaries section quite a bit.

"as the HDLSS datasets lack natural graph structure": GCN may be a solution
looking for a problem: it would be better to find datasets that actually have graph
structure.

*9: Justification of your recommendation
  Paper explores higher order interactions (3 samples) and is innovative and
has somewhat good results. Needs to tighten up the math and shorten it not
showing each gory detail (gets in the way). Also needs to do ablation experiments
to show gain over pairwise similarities.

*10: Three strong points of this paper (please numbwer each point)
  1: Results are impressive.
2: Explanation is pretty good but still missing some definitions. A few times
symbols are used before definition.
3:

*11: Three weak points of this paper (please number each point)
  1: "as the HDLSS datasets lack natural graph structure": GCN may be a solution
looking for a problem: it would be better to find datasets that actually have graph
structure.
2: Terms are used before definition eg in Spectral Graph Convolution section. It
would also be good to motivate some definitions rather than only presenting them
and include at least one intuitive example.
3: Would have liked more direct experiments: to compare order-2 versus
order-3 tensors in your own approach.

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [_] 3 Yes
  [X] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Paper
  [X] Short Paper
  
*19: Detailed comments for the authors
  Break up intro section into intro and previous work.

Can one not get higher order by iteratively doing pairwise combinations?

"Indeed, this also indicates that the structural feature extracted
from decomposable high-order similarity is inherently deter- mined by
pairwise similarity, which holds true for pairwise Laplacian matrix
and its eigenvectors." ... in which case using pairwise by itself may not be sufficient?

When presenting tensor indices (frontal v lateral v horizontal), it would be useful
if you could connect those to the familiar row and column axes: which correspond
to which? This would be pedagogically useful, even if all mappings would be
mathematically equivalent. For instance, T(i, :, :) [horizontal] may be taken to be
row indices. And T(:, i, :) [lateral] to be column indices. And those first two define
a matrix and the third extends "into the page" [front to back].

In "Revisit Spectral": presumably X is after Fourier has been applied? If so, pls
state it.

For similarity matrix: what are some examples? (you need to mention it here),
not later.

Eq 6: Pls remind readers what applying the eigenvectors U(.) does and give some
examples of what form the kernel g(.) could take. Eq. 6 is key and needs more
motivation. Is lower case x in eq 6 the data matrix X ? Then use $X$ instead of
lower case : it is confusing.

It would be cleaner if you could define $\Lambda$ as a vector e.g.
$(\lambda_1, .. \lambda_n)$ and then state that applying g(.) to it produces a
diagonal matrix. Lambda was referenced before definition.

What kind of rescaling produces \hat{Lambda} from $Lambda$ ?

Eq7: one k corresponds to one polynomial implying beta_k is a single scalar but
how can that be given that each Chebyshev polynomial carries several scalar
coefficients? If beta_k is a vector, pls state it.

Eq8: ignores the $S_{ik} interaction!?

$kNN$: not defined. Is it k-nearest neighbor?

Section IVB: Say what algo the runner-up wrt to which you achieved an
improved accuracy. Also : is the improvement statistically significant?

Eq12: In writing there is Y_1, Y_2 and Y_3 but in eq12 there is Y_0, Y_1 and Y_2.

Table II: Do not show 5 significant digits: it is visually polluting and you do not
have enough data to support it: if instances are O(100), you can only support 2
significant digits.

typos: run your paper thru a Grammar checker:
chebyshev -> Chebyshev
Hadamand -> Hadamard
repeatly -> repeatedly
recurisively
muilt-order
"as the pairwise similarity similarity does"


========================================================
The review report from reviewer #2:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [X] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High)
  [_] 1 (Medium)
  [X] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [X] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  The primary contribution of this paper is the integration of traditional graph information with high-order tensor similarity, enhancing the model’s ability to capture complex relationships within data. This model is specifically engineered to boost predictive accuracy and robustness in high-dimensional, low-sample-size (HDLSS) data settings.

*9: Justification of your recommendation
  The proposed method integrates high-order information to deal with HDLSS data and achieve more robust prediction. The model performance is much better than the baselines.

*10: Three strong points of this paper (please number each point)
  i) The introduction of high-order tensor similarity allows the Tensor-GCN to capture complex inter-sample relationships that go beyond traditional pairwise interactions. ii) Empirical results affirm that the Tensor-GCN consistently outperforms existing models. iii) The effectiveness of the method is validated across various scenarios.

*11: Three weak points of this paper (please number each point)
  i) The equation and reference numbers in the paper are not properly configured for hyperlinked navigation ii) The datasests used in the study are not correctly referenced. iii) The study is limited to biological datasets, which might restrict the generalizability of the findings to other domains.

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Paper
  [X] Short Paper
  
*19: Detailed comments for the authors
  The paper presents an innovative approach to enhancing semi-supervised classification in HDLSS data settings through the integration of high-order tensor similarity with traditional graph information. However, the technical writing could be further improved by adding hyperlinked navigation and dataset citation.

========================================================
The review report from reviewer #3:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [_] 3 (High)
  [X] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [X] 2 (High)
  [_] 1 (Medium)
  [_] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [X] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  Based on the formulation of polynomial-based spectral GNN, this work proposes Tensor-GCN, which mainly includes a modification on the original Laplacian matrix and higher-order structure by a pairwise similarity matrix and a higher-order similarity matrix.

*9: Justification of your recommendation
  The novelty, the solidity of the literature review, and the scope of the experiments are limited, so I recommend rejection. See my points below.

*10: Three strong points of this paper (please number each point)
  1.The writing and organization of the paper is satisfactory and easy to follow.
2. Experiments demonstrate the effectiveness of the proposal.
3. The mathematical presentation is generally clear.

*11: Three weak points of this paper (please number each point)
  1. More explanation is needed about why high-dimensional data is a problem for graph representation learning. Since some theoretical work [1] has talked about the relationship between the dimensionality of the sample size (#nodes) and features, where higher dimensionality comes with better model expressiveness ability, it is necessary for the authors to explain their arguments more.
2. The literature review is not comprehensive, some main arguments can be easily rejected. For the approximation ability of different polynomials, how is the sentence "Increasing the order of the polynomials filter directly may even inadvertently amplify this error, since an L2 term was added earlier" supported by the ChebyNetII work? Because different polynomial bases have very different statistical properties.

The proposal is limited to homophilic graphs. 

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [_] 3 Yes
  [X] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Paper
  [X] Short Paper
  
*19: Detailed comments for the authors
  1. Major:
 1. More explanation is needed about why high-dimensional data is a problem for graph representation learning. Since some theoretical work [1] has talked about the relationship between the dimensionality of the sample size (#nodes) and features, where higher dimensionality comes with better model expressiveness ability, it is necessary for the authors to explain their arguments more.
 2. The literature review is not comprehensive, some main arguments can be easily rejected. For the approximation ability of different polynomials, how is the sentence "Increasing the order of the polynomials filter directly may even inadvertently amplify this error, since an L2 term was added earlier" supported by the ChebyNetII work? Because different polynomial bases have very different statistical properties.
 3. As follows from the first point, the experimental comparisons are not state of the art. For example, ChebNetII [35] is a modified version of ChebNet [32], and a general performance improvement is achieved; however, ChebNetII is not included in the experiments. As well as spectral GNN BernNet and other spatial models.
 4. The proposal is limited to homophilic graphs.
2. Minor:
 1. Typos:
 1. The mathematical notation for node features are not consistent, e.g., normal x and \textit x are both used in equation 12.
 2. Ambiguous notations:
 1. What does it mean by $\times_2$ and $\times_3$ in equation 2?
3. Questions:
 1. What does the sentence below Equation 11 mean, "Equation (11) can capture more neighborhood information than the spectral filter.
 2. What's the relationship between the proposal and other hypergraph-based GNNs in terms of theory and formulation?

========================================================
The review report from reviewer #4:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [X] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [X] 3 (Good)
  [_] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High)
  [X] 1 (Medium)
  [_] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [X] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  The paper proposes a graph spectral kernel that takes into account third-order information as opposed to just pairwise. This kernel is then used for graph convolution networks (GCN). 

The definition is equation (12), which consists of concatenating the usual Laplacian kernel with a Laplacian tensor. The paper gave one example of such Laplacian tensor inspired by Cai et al, equation (15). This proposed tensor gives "triple comparisons" that is cosine-similarity-like, and could spur more theoretical studies since the form seems quite tractable. 

The rest of the paper proposes a specific network architecture that uses this graph spectral kernel for semi-supervised tasks, then implements this and tests on benchmark datasets. 

It is not surprising that the new method performs better as the kernel is of higher order. It would be interesting to have an ablation study where the same architecture is kept but a simpler kernel is used to validate the added value of the tensor kernel. The lack of such a study is a major weakness of this paper.

*9: Justification of your recommendation
  This paper would be useful to practitioners with moderately-sized graphs (~ 1e5 nodes) who want to try higher-order kernels. However, the lack of a good ablation study makes it hard to tell if the extra computational cost incurred by the 3-tensor is worth it.

*10: Three strong points of this paper (please number each point)
  1. Clear presentation and decent literature review. 
2. Datasets provided
3. New 3-way tensor with a simple form that could spur interesting theoretical studies.

*11: Three weak points of this paper (please number each point)
  1. Lack of an ablation study.
2. Regarding runtime, the kernel is effectively a matrix of dimension n^2 x n, and the tested datasets are quite small (~1e4 to 1e5 features). So I expect this to be useful for datasets of this size or below, but not much larger. 
3. Codes are not provided at https://anonymous.4open.science/r/High-Dimensional-Low-Sample-Size-
78E8 , only the datasets?

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Paper
  [X] Short Paper
  
*19: Detailed comments for the authors
  I particularly would like to know how much value does the 3-tensor add. For example, how would the experimental results look like if we didn't have the 3-tensor, only the Laplacian? How does this compare to graph diffusion methods, in the most controlled way possible? The simplest ablation I can think of is to remove the tensor component of equation (12) and keep the later part of the network the same. Another variation is to replace the $k=2$ part of the equation (12) with the second-power of the transition matrix to have a direct comparison with DCNNs. 

Given that 3-tensor is a computational burden, I think it would be helpful to demonstrate when the 3-tensor does add significant values. 

========================================================
The review report from reviewer #5:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [_] 3 (Innovative)
  [X] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [_] 3 (High)
  [_] -2 (Marginal)
  [X] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [_] 3 (Good)
  [X] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [_] 3 (Yes)
  [X] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [_] 2 (High)
  [X] 1 (Medium)
  [_] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [_] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [X] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  The paper introduces Tensor Graph Convolutional Networks (Tensor-GCN), which leverage high-order tensor similarity to improve semi-supervised classification performance on high-dimensional, low-sample size (HDLSS) data.

*9: Justification of your recommendation
  It is an interesting idea about Tensor-GCN approach for HDLSS data classification but would benefit from additional theoretical analysis and discussion of computational complexity to strengthen its contribution.

*10: Three strong points of this paper (please number each point)
  1. It is an interesting idea to include Tensor-GCN for the task.
2. The model demonstrates promising performance in the experiments.
3. The paper includes t-SNE visualizations and ROC curves that illustrate the performance of Tensor-GCN over baseline methods. These visualizations provide an intuitive understanding.

*11: Three weak points of this paper (please number each point)
  1. The paper focuses heavily on empirical results and comparisons but lacks in-depth theoretical analysis of why the proposed Tensor-GCN method works better for HDLSS data. More rigorous justification for the advantages of using tensor-based similarity and the proposed architecture could strengthen the paper.
2. While the paper briefly mentions increased runtime compared to standard GCN, there is no detailed analysis of the computational complexity of Tensor-GCN. Given that it uses higher-order tensor operations, a more thorough discussion of computational costs and scalability to larger datasets would be beneficial. Moreover, the runtime here is not very convincing as it may be influenced by machine status or other processes running. A more objective metric should be included, e.g. FLOPs.
3. The paper does not include an ablation study or analysis of how sensitive Tensor-GCN's performance is too different hyperparameters like the number of layers, hidden dimensions etc. Understanding how robust the method is to parameter choices would provide more confidence in its practical applicability.

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [X] No
  [_] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [X] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [X] Regular Paper
  [_] Short Paper
  
*19: Detailed comments for the authors
  Please refer to the weak points.

========================================================
The review report from reviewer #6:

*1: Is the paper relevant to ICDM?
  [_] No
  [X] Yes
  
*2: How innovative is the paper?
  [_] 6 (Very innovative)
  [X] 3 (Innovative)
  [_] -2 (Marginally)
  [_] -4 (Not very much)
  [_] -6 (Not at all)
  
*3: How would you rate the technical quality of the paper?
  [_] 6 (Very high)
  [X] 3 (High)
  [_] -2 (Marginal)
  [_] -4 (Low)
  [_] -6 (Very low)
  
*4: How is the presentation?
  [_] 6 (Excellent)
  [_] 3 (Good)
  [X] -2 (Marginal)
  [_] -4 (Below average)
  [_] -6 (Poor)
  
*5: Is the paper of interest to ICDM users and practitioners?
  [X] 3 (Yes)
  [_] 2 (May be)
  [_] 1 (No)
  [_] 0 (Not applicable)
  
*6: What is your confidence in your review of this paper?
  [X] 2 (High)
  [_] 1 (Medium)
  [_] 0 (Low)
  
*7: Overall recommendation
  [_] 6: must accept (in top 25% of ICDM accepted papers)
  [X] 3: should accept (in top 80% of ICDM accepted papers)
  [_] -2: marginal (in bottom 20% of ICDM accepted papers)
  [_] -4: should reject (below acceptance bar)
  [_] -6: must reject (unacceptable: too weak, incomplete, or wrong)
  
*8: Summary of the paper's main contribution and impact
  The paper proposes high-order tensor similarity to describe relationships
among multiple samples, aiming to extract more valuable information
from the graph structure. Based on this, the authors develop 
the Tensor-based Graph Convolutional Network (Tensor-GCN) that effectively integrates traditional graph information with high-order graph information. By employing a multi-layer Tensor-GCN framework, traditional pairwise information with high-order neighborhood information is seamlessly integrated,
thus achieving more accurate and robust predictive capabilities.

*9: Justification of your recommendation
  The paper proposes a novel graph neural network designed for high-dimensional, low-sample size datasets, addressing a significant real-world problem that will interest the ICDM community. Considering there are some drawbacks in the content and length of the paper, I recommend accepting it as a short paper.

*10: Three strong points of this paper (please number each point)
  1. This paper addresses the common real-world issue of high-dimensional, low-sample size (HDLSS) datasets. The author effectively illustrates the associated problems and challenges, providing comprehensive background knowledge.
2. Inspired by tensor spectrum analysis, the author proposes Tensor-GCN, a novel approach capable of capturing high-order information.
3. The numerical experiments demonstrate the potential of the proposed Tensor-GCN on HDLSS datasets.

*11: Three weak points of this paper (please number each point)
  1. The justification and descriptions for the proposed Tensor-GCN needs to be strengthened.
2. Additional experiments or discussions are necessary to thoroughly evaluate the proposed method.
3. There are some inaccuracies in the writing that need proofreading and improvement.

*12: Is this submission among the best 10% of submissions that you reviewed for ICDM'24?
  [_] No
  [X] Yes
  
*13: Are the datasets used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*14: If the authors use private data in the experiments, will they publish data for public access in the camera-ready version of the paper?
  [_] 3 Yes
  [_] 2 Partial
  [X] 1 No
  [_] 0 Not applicable
  
*15: Are the competing methods used in the study correctly identified and referenced?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*16: Will the authors publish their source code for public access in the camera-ready version of the paper?
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*17: Is the experimental design detailed enough to allow for reproducibility? (You can also include comments on reproducibility in the body of your review.)
  [X] 3 Yes
  [_] 2 Partial
  [_] 1 No
  [_] 0 Not applicable
  
*18: If the paper is accepted, which format would you suggest?
  [_] Regular Paper
  [X] Short Paper
  
*19: Detailed comments for the authors
  (1a) It is unclear why the author picks the third order for Tensor-GCN. Is it possible to extend to a higher order, and what are the trade-offs?
(1b) I find the bottom row of Fig. 1 difficult to follow. For example, what does "e_{123}" mean? The author should highlight and explain the figure, linking the text in Sec 3 back to it.

(2a) What is the performance of Tensor-GCN on HDLSS datasets in domains other than biological? Additionally, I am also curious about its performance on non-HDLSS datasets.
(2b) Memory utilization should also be discussed in Table 3.
(3c) Most of the baselines are over five years old and outdated. I am interested in seeing comparisons with papers [23] and [25] mentioned in the introduction.

(3a) The value of k in Formula (12) is confusing. It should start from k=1 instead of k=0 to maintain consistency with the text.
(3b) In the experiment settings, the paper states, "Samples are partitioned into labeled and unlabeled sets at an 8:2 ratio, with 20% of the labeled samples" should be modified to “labeled and unlabeled sets at a 2:8 ratio”.
(3c) I suggest accepting it as a short paper. The author can condense some of the verbose descriptions in the preliminaries and baselines sections.


========================================================
```
