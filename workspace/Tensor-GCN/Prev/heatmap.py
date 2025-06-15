import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import tensorly as tl
import torch.nn.functional as F

# 注意，这个地方算的不是 L 和 L3，而是 T 和 T3

save_dir ='/home/wangcy/sandbox/Tensor-GCN_Claude/heatmap_pig'
mat_folder = '/home/wangcy/sandbox/Tensor-GCN_Claude/ZDataSet'
dataset = 'GMM_data_300'
# dataset = 'GMM_data_600'

# 定义距离计算函数
def disT(X):
    """
    使用欧氏距离计算样本之间的距离。
    参数：
        X: numpy 数组，形状为 (N, D)
    返回：
        距离矩阵，形状为 (N, N)
    """
    return pairwise_distances(X)


def compute_tensor_T2(X, sigma=None):
    """
    计算基于距离平方的权重矩阵 T2（即二阶相似度矩阵），
    使用高斯核函数： T2(i,j) = exp(-||xi - xj||^2/(2*sigma^2))
    
    参数：
        X: 输入数据矩阵，形状为 (N, D)，必须为 numpy 数组
        sigma: 距离衰减参数。如果为 None，则自动以上三角非对角线距离中位数为 sigma。
    返回：
        T2: 二阶相似度矩阵，PyTorch 张量，形状为 (N, N)
    """
    start = time.time()
    dist = disT(X)         # 计算欧氏距离矩阵
    dist_squared = dist ** 2

    dist_vec = dist.reshape(1, X.shape[0]*X.shape[0])
    sigma = np.mean(dist_vec)
    if sigma is None:
        # 提取上三角（不含对角线）距离
        distances = dist[np.triu_indices_from(dist, k=1)]
        sigma = np.median(distances)
    T = np.exp(-dist_squared / (2 * sigma ** 2))
    T = torch.tensor(T, dtype=torch.float)
    end = time.time()
    # print('Time for T2 computation:', end - start)
    return T


def compute_T3(X, feat_cog=1, undecomposable=True):
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
    similarity = Eye - (dot_product / (norm_ij * norm_jk))

    return similarity

    L3 = similarity
    L3 = F.normalize(similarity, p=2, dim=-1) 

    L3 = L3.reshape(N**2, N)
    d = torch.sum(L3, dim = 0)# ∑T_3 :i
    D_31 = torch.diag(torch.pow(torch.kron(d,d), -0.5))
    D_32 = torch.diag(torch.pow(d, -0.5))
    L3 = D_31@L3@D_32
    return L3


def compute_dec_T3(X):
    T2 = compute_tensor_T2(X)
    return tl.tenalg.khatri_rao([T2,T2]).T


# not used
def triad_affinity(X): 
    """
    计算三阶相似度张量 T3，形状为 (N, N, N)。
    示例实现：先计算二阶相似度 T2（采用与 compute_tensor_T2 类似的方法），
    然后定义 T3(i,j,k) = T2(i,j) * T2(i,k) * T2(j,k)
    
    参数：
        X: 输入数据矩阵，形状为 (N, D)，可以是 numpy 数组或 torch.Tensor
    返回：
        T3: 三阶相似度张量，PyTorch 张量，形状为 (N, N, N)
    """
    # 保证 X 为 numpy 数组
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().detach().numpy()
    else:
        X_np = X

    # 计算二阶相似度 T2
    T2 = pairwise_distances(X_np)
    distances = T2[np.triu_indices_from(T2, k=1)]
    sigma = np.median(distances)
    T2 = np.exp(-T2**2/(2*sigma**2))
    N = T2.shape[0]
    T3 = np.zeros((N, N, N))
    # 简单实现：T3(i,j,k) = T2(i,j) * T2(i,k) * T2(j,k)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                T3[i, j, k] = T2[i, j] * T2[i, k] * T2[j, k]
    T3 = torch.tensor(T3, dtype=torch.float)
    return T3

def visualize_raw_data(X, save_path=None, cmap='summer'):
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(X, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')
    plt.title('Raw Data Matrix', fontsize=16, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600)
    plt.show()

def visualize_dyadic_heatmap(T2, save_path=None, cmap='summer'):
    if isinstance(T2, torch.Tensor):
        T2 = T2.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(T2, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')
    plt.title('Dyadic Similarity Matrix', fontsize=16, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600)
    plt.show()

def visualize_triad_hotmap(T3, save_path=None, cmap='summer'):
    if isinstance(T3, torch.Tensor):
        T3 = T3.cpu().detach().numpy()
    
    N = T3.shape[0]
    T3_unfolded = T3.reshape(N * N, N)
    
    plt.figure(figsize=(14, 12))
    plt.imshow(T3_unfolded, aspect='auto', cmap=cmap, interpolation='nearest', origin='lower')
    plt.title('Triad Similarity Matrix', fontsize=16, pad=20)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600)
    plt.show()

def normalize(X):
    # 使用MinMaxScaler将数据归一化到[0,1]区间
    scaler = MinMaxScaler()
    X_flat = X.reshape(-1, 1)
    X_normalized = scaler.fit_transform(X_flat).reshape(X.shape)
    X = X_normalized
    return X


def main():
    # 假设文件夹中有一个 .mat 文件，比如 'data.mat'
    mat_file = os.path.join(mat_folder, f'{dataset}')

    # 加载 .mat 文件（假设里面有变量 'X'）
    mat_data = scipy.io.loadmat(mat_file)
    X = mat_data['data']
    print('Loaded data with shape:', X.shape)

    # 确保 X 为 float32 类型
    X = X.astype(np.float32)
    # 可视化原始数据
    X_normalized = normalize(X)
    
    # 计算二阶相似度 T2
    T2 = compute_tensor_T2(X)
    # 计算三阶相似度 T3
    # T3 = triad_affinity(X)
    T3_dec = compute_dec_T3(X)
    T3 = compute_T3(X)

    T2 = normalize(T2)
    T3_dec = normalize(T3_dec)
    T3 = normalize(T3)

    visualize_raw_data(X_normalized, save_path=os.path.join(save_dir, f'{dataset}_raw.png'), cmap='RdYlBu_r')
    # 可视化二阶相似度热图，并保存到指定路径
    visualize_dyadic_heatmap(T2, save_path=os.path.join(save_dir, f'{dataset}_2.png'), cmap='RdYlBu_r')
    # 可视化三阶相似度热图，并保存到指定路径
    visualize_triad_hotmap(T3_dec, save_path=os.path.join(save_dir, f'{dataset}_3(Dec).png'), cmap='RdYlBu_r')
    visualize_triad_hotmap(T3, save_path=os.path.join(save_dir, f'{dataset}_3.png'), cmap='RdYlBu_r')

if __name__ == '__main__':
    main()

    # 'viridis'







    # plt.figure(figsize=(8, 6))
    # plt.imshow(T2, aspect='auto', cmap='hot')
    # plt.title(f'Dyadic Similarity')
    # plt.xlabel('Samples')
    # plt.ylabel('Samples')
    # plt.colorbar()


    # plt.figure(figsize=(10, 8))
    # plt.imshow(T3_unfolded, aspect='auto', cmap='viridis')
    # plt.title(f'Triad Similarity')
    # plt.xlabel('Dimension')
    # plt.ylabel('Unfolded Samples (N*N)')
    # plt.colorbar()