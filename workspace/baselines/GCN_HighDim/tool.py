import numpy as np
import torch
from torch import nn
from scipy import linalg
import scipy.sparse as sp
import tensorly as tl
from scipy.sparse.linalg import eigsh
import pickle as pkl
import time

# 全局定义
device = torch.device("cpu")
tl.set_backend('pytorch')

#------------
# 功能函数
#------------
# 预处理 adj和features
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float64)  # 每个node所有通道求和。用于feature归一化
    # rowsum = np.array(features.sum(1)) 
    r_inv = np.power(rowsum, -1).flatten()  # 1除以rowsum，再flatten
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)  # features * diag(r_inv), 是矩阵乘法，不是element wise
    return torch.from_numpy(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    ## 把D^-0.5 A~ D^-0.5 计算完毕
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))  # 直接把A+In传入再normalize，就和论文里的A^一样了
    return adj_normalized
