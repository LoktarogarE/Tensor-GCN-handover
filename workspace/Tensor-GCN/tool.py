import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import linalg
import scipy.sparse as sp
import tensorly as tl
from tensorly.tenalg import mode_dot
# from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh

import pickle as pkl
import time

# 全局定义
device = torch.device("cuda:3")
tl.set_backend('pytorch')
is_cheb = False   # 这里千万记得修改！


#------------
# 功能函数
#------------
# 预处理 adj和features
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float64)  # 每个node所有通道求和。用于feature归一化
    r_inv = np.power(rowsum, -1).flatten()  # 1除以rowsum，再flatten
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)  # features * diag(r_inv), 是矩阵乘法，不是element wise
    return torch.from_numpy(features)

    # feat = torch.from_numpy(features)
    # norm = torch.norm(feat, p=2, dim=1, keepdim=True)
    # feat_norm = feat / norm
    # return feat_norm

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
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))  
    return torch.from_numpy(adj_normalized)

# for heatmap test
def compute_T3(X, feat_cog=1e-9, undecomposable=True):
    # 将输入张量转换为浮点型
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    dist = torch.cdist(X, X)**2
    X = dist.float()
    N = X.shape[0]
    
    # 计算所有可能的差值对
    # 扩展维度以便广播
    X_i = X.unsqueeze(1).unsqueeze(2)  # Shape: [N, 1, 1, D]
    X_j = X.unsqueeze(0).unsqueeze(2)  # Shape: [1, N, 1, D]
    X_k = X.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, N, D]
    
    # 计算差值
    diff_ij = X_i - X_j  # Shape: [N, N, 1, D]
    diff_jk = X_k - X_j  # Shape: [1, N, N, D]
    
    # 计算范数
    norm_ij = torch.norm(diff_ij, dim=-1)  # Shape: [N, N, 1]
    norm_jk = torch.norm(diff_jk, dim=-1)  # Shape: [1, N, N]
    
    # 计算点积
    dot_product = torch.sum(diff_ij * diff_jk, dim=-1)  # Shape: [N, N, N]
    
    # 计算相似度
    Eye = torch.ones(N, N, N)
    # similarity = Eye - (dot_product / (norm_ij * norm_jk + feat_cog))
    similarity = Eye - (dot_product / (norm_ij * norm_jk + feat_cog))

    return similarity.reshape(N*N, N)





def get_L3(X,feat_cog=1e-9, undecomposable=True):
    # 将输入张量转换为浮点型
    dist = torch.cdist(X, X)**2
    X = dist.float()
    N = X.shape[0]
    # 计算样本之间的差值
    diff_ij = (X[:, None] - X[None, :])
    diff_jk = (X[None, :] - X[:, None])

    # 计算样本之间的距离
    dist_ij = torch.linalg.norm(diff_ij, dim=-1)
    dist_jk = torch.linalg.norm(diff_jk, dim=-1)
    # 计算差值的点积
    dot_product = torch.einsum('ijk,ikl->ijk', diff_ij, diff_jk)  
    print('dot:', dot_product)
    # 计算三阶相似性张量
    # Eye = torch.eye(N).unsqueeze(0).repeat(N, 1, 1)
    Eye = torch.ones(N,N,N)
    # similarity = Eye - (dot_product / (dist_ij[:, :, None] * dist_jk[None, :, :]+1e-9))
    similarity = Eye - (dot_product / (dist_ij[:, :, None] * dist_jk[None, :, :]+feat_cog))   

    # print("Sim:", similarity)
    L3 = similarity
    L3 = F.normalize(similarity, p=2, dim=-1) 

    L3 = L3.reshape(N**2, N)
    d = torch.sum(L3, dim = 0)# ∑T_3 :i
    # print("Degree:", d)
    D_31 = torch.diag(torch.pow(torch.kron(d,d), -0.5))
    D_32 = torch.diag(torch.pow(d, -0.5))
    L3 = D_31@L3@D_32
    
    # L3 = L3.reshape(N, N, N) #MTGCN
    # return F.normalize(L3, p=2, dim=1)
    print("Sim:", L3)
    return L3

def get_L3_tqdm(X,feat_cog, undecomposable=True):
    """
    计算三阶相似性张量 L3。

    参数：
    - X (torch.Tensor): 输入特征张量，形状为 (N, D)。
    - feat_cog (float): 一个小的常数，用于防止除零错误。
    - undecomposable (bool): 是否使用不可分解相似度。

    返回：
    - L3 (torch.Tensor): 计算得到的三阶相似性张量，形状为 (N, N, N)。
    """
    # 将输入张量转换为浮点型并计算距离
    print("计算距离...")
    dist = torch.cdist(X, X)**2
    X = dist.float()
    N = X.shape[0]
    
    # 初始化差值张量
    print("计算样本之间的差值...")
    diff_ij = torch.empty(N, N, X.shape[1], device=X.device)
    diff_jk = torch.empty(N, N, X.shape[1], device=X.device)
    
    for i in tqdm(range(N), desc="差值计算"):
        diff_ij[i] = X[i].unsqueeze(0) - X
        diff_jk[i] = X - X[i].unsqueeze(0)
    
    # 计算样本之间的距离
    print("计算样本之间的距离...")
    dist_ij = torch.empty(N, N, device=X.device)
    dist_jk = torch.empty(N, N, device=X.device)
    
    for i in tqdm(range(N), desc="距离计算"):
        dist_ij[i] = torch.linalg.norm(diff_ij[i], dim=-1)
        dist_jk[i] = torch.linalg.norm(diff_jk[i], dim=-1)
    
    # 计算差值的点积
    print("计算差值的点积...")
    dot_product = torch.empty(N, N, N, device=X.device)
    
    for i in tqdm(range(N), desc="点积计算"):
        for j in range(N):
            dot_product[i, j] = torch.dot(diff_ij[i, j], diff_jk[i, j])
    
    # 计算三阶相似性张量
    print("计算三阶相似性张量...")
    similarity = torch.ones(N, N, N, device=X.device)
    
    for i in tqdm(range(N), desc="相似性计算"):
        for j in range(N):
            for k in range(N):
                similarity[i, j, k] -= dot_product[i, j, k] / (dist_ij[i, j] * dist_jk[j, k] + feat_cog)
    
    # 归一化
    print("归一化相似性张量...")
    T = F.normalize(similarity, p=2, dim=-1)
    T = T.reshape(N**2, N)
    print("Sim:", T)
    
    # 计算度
    print("计算度...")
    d = torch.sum(T, dim=0)  # ∑T_3 :i
    print("Degree:", d)
    
    # 计算 D_31 和 D_32
    print("计算 D_31 和 D_32...")
    D_31 = torch.diag(torch.pow(torch.kron(d, d), -0.5))
    D_32 = torch.diag(torch.pow(d, -0.5))
    
    # 计算 L3
    print("计算 L3...")
    L3 = D_31 @ T @ D_32
    L3 = L3.reshape(N, N, N)
    print("L3 计算完成。")
    
    return L3


# 实现大矩阵
def get_support(H, L, L3):
    H = H.to(device)
    L = L.to(device)
    L3 = L3.to(device)
    B2 = torch.mm(L,H).to(device)
    if is_cheb == True:
        B3 = (2*L@L@H)-H
    else:
        H_kha = tl.tenalg.khatri_rao([H,H])
        B3 = torch.zeros(L.shape[0], H_kha.shape[1])
        B3 = L3.matmul(H_kha) - H
    B = torch.hstack((H, B2, B3)) # 全用
  # B = B3  # 只用3阶
  # B = torch.hstack((H,B3))  # 只用 1 3 阶
  # B = torch.hstack((B2, B3)) # 只用 2 3 阶
    
    return B

# def get_support(H, L, L3): # for MTGCN para test
#     N = H.shape[0]
#     H = H.to(device)
#     L = L.to(device)
#     L3 = L3.to(device)
#     L3 = L3.reshape(N, N, N)
#     B2 = torch.mm(L,H).to(device)
#     B3 = compute_Y(L3, H, method='sum', weight_matrix=None).to(device)


    B = torch.hstack((H, B2, B3)) # 全用
    
    return B

def get_support_MTGCN(H, L, L3):
    H = H.to(device)
    L = L.to(device)
    L3 = L3.to(device)
    B2 = torch.mm(L,H).to(device)
    B3 = compute_Y(L3, H, method='sum', weight_matrix=None).to(device)
    assert H.shape == B2.shape == B3.shape

    # print("B2", B2)
    # print("B3", B3)
    return H + B2 + B3

def compute_Y(L3, X, method='sum', weight_matrix=None):
    """
    计算 Y = L3 ×₃ X^T ×₂ X^T 并压缩第三维度

    参数:
    - L3: 一个形状为 (N, N, N) 的3阶张量
    - X: 一个形状为 (N, D) 的矩阵
    - method: 压缩方法，支持 'sum', 'mean', 'max', 'matmul'。默认是 'sum'
    - weight_matrix: 如果 method='matmul'，需要提供一个形状为 (D, D) 的权重矩阵

    返回:
    - Y: 计算得到的矩阵，形状为 (N, D)
    """
    # 计算 X 的转置
    X_t = X.T  # 形状: (D, N)
    
    # 第一次 mode-3 乘积: L3 ×₃ X^T
    temp = mode_dot(L3, X_t, mode=2)  # 形状: (N, N, D)
    
    # 第二次 mode-2 乘积: (L3 ×₃ X^T) ×₂ X^T
    Y3 = mode_dot(temp, X_t, mode=1)  # 形状: (N, D, D)
    
    # 压缩第三维度
    if method == 'sum':
        Y = tl.sum(Y3, axis=2)  # 形状: (N, D)
    elif method == 'mean':
        Y = tl.mean(Y3, axis=2)  # 形状: (N, D)
    elif method == 'max':
        Y = tl.max(Y3, axis=2)   # 形状: (N, D)
    elif method == 'matmul':
        if weight_matrix is None:
            raise ValueError("weight_matrix must be provided when method='matmul'")
        # 确保 weight_matrix 的形状为 (D, D)
        if weight_matrix.shape != (Y3.shape[2], Y3.shape[2]):
            raise ValueError(f"weight_matrix 的形状必须为 ({Y3.shape[2]}, {Y3.shape[2]})")
        # 使用矩阵乘法压缩第三维度
        Y = tl.tensordot(Y3, weight_matrix, axes=([2], [0]))  # 形状: (N, D)
    else:
        raise ValueError("Unsupported compression method. Choose from 'sum', 'mean', 'max', 'matmul'")
    
    return Y


def sparse_khatri_rao_product(sparse_mat1, sparse_mat2):
    # 获取稀疏矩阵的非零元素索引和值
    indices1 = sparse_mat1._indices()
    values1 = sparse_mat1._values()
    indices2 = sparse_mat2._indices()
    values2 = sparse_mat2._values()
    
    # 获取稀疏矩阵的形状
    m1, n1 = sparse_mat1.size()
    m2, n2 = sparse_mat2.size()
    
    # 计算结果矩阵的维度
    result_shape = (m1 * m2, n1)
    
    # 计算结果矩阵的非零元素索引和值
    result_indices = torch.cat([indices1.repeat(1, m2), indices2.repeat(m1, 1)], dim=1)
    result_values = torch.outer(values1, values2).reshape(-1)
    
    # 创建结果稀疏矩阵
    result_sparse_mat = torch.sparse_coo_tensor(result_indices, result_values, result_shape)
    
    return result_sparse_mat
