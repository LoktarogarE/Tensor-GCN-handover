# To do

1. ## 注意GCN 中的 Relu 去掉了
2. KNN 修改 尝试(on Lung/Gli85) 超图使用10~15近邻
3. 模型写法 要改, 要么想个别的故事, 尽量避开从ChebyNet来推理
4. TGCN拉普拉斯修订检查
5. 尝试把三个\theta进行统一
6. 统计方法修订，pr, auc 等统计的范围应当缩小
7. 结果保存
   1. 最优层的pth文件
8. UTC的相似度实际上是cos

一个2.4G内存，3000mb显存



# 寄了的数据集

CLL ｜ SMK ｜ GLIOMA ｜ Colon ？ 还未确认

degree 的波动应该在 100 以上



leuk Final test  ACC: 87.931 %  0.8793103694915771, F:0.8998, Pr:0.9028, Auc:0.8694, Rec:0.9028

ALL 





---

The output of g*x is another Nx1 vector, which represents the result of applying the graph convolution to the input vector x. The graph convolution is a linear transformation that operates on the graph Laplacian matrix and the filter parameter theta. The graph convolution can be seen as a way of smoothing or aggregating the features of the neighboring nodes in the graph, weighted by the filter parameter and the graph structure





nohup python GCN_search.py > log/smk.log &

nohup python GCN_search.py > log/all.log &

nohup python GCN_search.py > log/all.log &

nohup python GCN_search.py > log/all.log &



TGCN(不可分解)

ALLAML  hd:64|dp:0.6|wd:0.05 || Layer 2, max_Acc:77.58621, F:0.8034, Pr:0.8194, Auc:0.7494, Rec:0.8194

GLI85 hd:256|dp:0.7|wd:0.005 || Layer 4, max_Acc:85.294122, F:0.8775, Pr:0.8824, Auc:0.8292, Rec:0.8824

Lung hd:512|dp:0.4|wd:0.0005 || Layer 3, max_Acc:86.885244, F:0.8471, Pr:0.867, Auc:0.9896, Rec:0.867

Pro hd:256|dp:0.4|wd:0.0005 || Layer 3, max_Acc:85.185188, F:0.8824, Pr:0.8824, Auc:0.8823, Rec:0.8824

TGCN(可分解)

ALLAML  hd:512|dp:0.5|wd:0.0005 || Layer 4, max_Acc:79.310346, F:0.7947, Pr:0.8056, Auc:0.7481, Rec:0.8056   

GLI85 hd:64|dp:0.4|wd:0.0005 || Layer 7, max_Acc:85.294116, F:0.8794, Pr:0.8824, Auc:0.84, Rec:0.8824

Lung hd:64|dp:0.5|wd:0.0005 || Layer 5, max_Acc:85.245895, F:0.796, Pr:0.8522, Auc:0.9535, Rec:0.8522

Pro hd:16|dp:0.5|wd:0.0005 || Layer 6, max_Acc:81.481487, F:0.852, Pr:0.8529, Auc:0.8546, Rec:0.8529

GCN

ALL hd:512|dp:0.8|wd:0.005 || Layer 2, max_Acc:75.862074, F:0.7859, Pr:0.8056, Auc:0.7294, Rec:0.8056

Gli85 hd:32|dp:0.7|wd:0.05 || Layer 4, max_Acc:82.352948, F:0.8478, Pr:0.8471, Auc:0.8253, Rec:0.8471

Lung hd:128|dp:0.4|wd:0.0005 || Layer 5, max_Acc:82.822084, F:0.8073, Pr:0.8374, Auc:0.9457, Rec:0.8374

Pro hd:16|dp:0.4|wd:0.0005 || Layer 7, max_Acc:82.716048, F:0.8626, Pr:0.8627, Auc:0.8623, Rec:0.8627











已解决问题：

ALL/CLL 相似度/高阶 graph 有负数问题

Colon Leuk 特征值有负数  XXX

统计范围修复，当前的

加入 ChebNet 的结果

### 半监督分类统计指标修复

> 去掉purity
>
> Recall-> 预测为正的样本在所有样本中的比例
>
> Precision->预测喂正的样本在所有正例样本中的比例
>
> 增加log-likelihood 



**为什么F-SCORE 等其他指标全为** **1** ？

# 结果

### ChebNet

ALLAML hd:8|dp:0.6|wd:0.05 || Layer 7, max_Acc:82.758617, F:1.0, Auc:1.0, Rec:1.0

GLI hd:512|dp:0.4|wd:0.0005 || Layer 4, max_Acc:82.352942, F:0.8849, Auc:0.9091, Rec:0.8824  #####

Leuk |dp:0.6|wd:0.0005 || Layer 6, max_Acc:89.655173, F:1.0, Auc:1.0, Rec:1.0

lung hd:8|dp:0.6|wd:0.0005 || Layer 5, max_Acc:79.141104, F:0.7311, Auc:0.971, Rec:0.8

pro hd:8|dp:0.7|wd:0.5 || Layer 2, max_Acc:65.432096, F:0.5916, Auc:0.6318, Rec:0.619                   

### GCN

ALLAML 

hd:8|dp:0.8|wd:0.0005 || Layer 6, max_Acc:82.758617, F:1.0, Auc:1.0, Rec:1.0
GLI 
 hd:16|dp:0.4|wd:0.05 || Layer 4, max_Acc:82.352942, F:0.942, Auc:0.9545, Rec:0.9412

Leuk

hd:16|dp:0.7|wd:0.0005 || Layer 6, max_Acc:89.655173, F:1.0, Auc:1.0, Rec:1.0



### TGCN DEC

ALLAML  hd:8|dp:0.7|wd:0.005 || Layer 3, max_Acc:79.310346, F:0.8464, Auc:0.8, Rec:0.8571

GLI hd:512|dp:0.5|wd:0.0005 || Layer 4, max_Acc:85.294122, F:1.0, Auc:1.0, Rec:1.0 

Leuk  hd:64|dp:0.8|wd:0.005 || Layer 4, max_Acc:89.655173, F:1.0, Auc:1.0, Rec:1.0

Lung hd:16|dp:0.6|wd:0.0005 || Layer 5, max_Acc:85.276067, F:0.8604, Pr:0.9, Auc:0.9668, Rec:0.9

pro hd:32|dp:0.5|wd:0.0005 || Layer 3, max_Acc:74.074078, F:1.0, Auc:1.0, Rec:1.0

### TGCN UTC

ALLAML hd:512|dp:0.6|wd:0.0005 || Layer 4, max_Acc:84.482753, F:0.8464, Auc:0.8, Rec:0.8571

GLI 

LEUK hd:8|dp:0.7|wd:0.0005 || Layer 4, max_Acc:89.655173, F:1.0, Auc:1.0, Rec:1.0

lung hd:128|dp:0.5|wd:5e-05 || Layer 3, max_Acc:95.705521, F:1.0, Auc:1.0, Rec:1.0

pro hd:8|dp:0.4|wd:0.0005 || Layer 4, max_Acc:76.543212, F:1.0, Auc:1.0, Rec:1.0



