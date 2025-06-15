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

save_dir = '/home/wangcy/sandbox/Tensor-GCN_Claude/embedding_pig'
pt_folder = '/home/wangcy/sandbox/Tensor-GCN_Claude/embedding'
dataset = 'SYN300'
# dataset = 'syn_data_300'


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


def visualize_emb(X, save_path=None, normalize=True):
    """
    可视化二阶相似度矩阵 T2 的热图。
    
    参数：
      T2: 二阶相似度矩阵，形状为 (N, N)
      save_path: 字符串，若提供则将图像保存到该路径（会自动创建目录）
      normalize: 布尔值，是否对数据进行归一化处理
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
    
    # 对数据进行归一化处理，确保所有图像的坐标轴尺度一致
    if normalize:
        # 使用MinMaxScaler将数据归一化到[0,1]区间
        scaler = MinMaxScaler()
        X_flat = X.reshape(-1, 1)
        X_normalized = scaler.fit_transform(X_flat).reshape(X.shape)
        X = X_normalized
    
    plt.figure(figsize=(12, 10))
    # 使用imshow绘制热图，aspect='auto'自动调整纵横比，使用viridis配色方案
    plt.imshow(X, aspect='auto', cmap='viridis', interpolation='nearest')
    # 设置图表标题，字体大小为16，上方留出20的填充空间
    plt.title('Embedding', fontsize=16, pad=20)
    # 设置x轴和y轴标签，字体大小为14
    # plt.xlabel('Sample Index', fontsize=14)
    # plt.ylabel('Sample Index', fontsize=14)
    # 设置刻度标签的字体大小为12
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar()
    # 设置颜色条刻度标签的字体大小为12，使其与图形整体风格保持一致
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Similarity Score', fontsize=14)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600) #bbox_inches='tight'
    plt.show()

def visualize_triad_hotmap(T3, save_path=None, normalize=True):
    """
    可视化三阶相似度 T3 的热图。
    
    参数：
        T3: 三阶相似度张量，形状为 (N, N, N)
        save_path: 字符串，如果提供，则保存图像到该路径。
        normalize: 布尔值，是否对数据进行归一化处理
    """
    if isinstance(T3, torch.Tensor):
        T3 = T3.cpu().detach().numpy()
    
    N = T3.shape[0]
    # 将 T3 展开为二维矩阵，形状为 (N*N, N)
    T3_unfolded = T3.reshape(N * N, N)
    
    # 对数据进行归一化处理，确保所有图像的坐标轴尺度一致
    if normalize:
        # 使用MinMaxScaler将数据归一化到[0,1]区间
        scaler = MinMaxScaler()
        T3_flat = T3_unfolded.reshape(-1, 1)
        T3_normalized = scaler.fit_transform(T3_flat).reshape(T3_unfolded.shape)
        T3_unfolded = T3_normalized
    
    plt.figure(figsize=(14, 12))
    plt.imshow(T3_unfolded, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.title('Triad Similarity Matrix', fontsize=16, pad=20)
    # plt.xlabel('Dimension', fontsize=14)
    # plt.ylabel('Unfolded Samples (N*N)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar()
    # 设置颜色条刻度标签的字体大小为12，使其与图形整体风格保持一致
    cbar.ax.tick_params(labelsize=12)
    # cbar.set_label('Similarity Score', fontsize=14)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600) #, bbox_inches='tight'
    plt.show()

def main():
    # 假设文件夹中有一个 .mat 文件，比如 'data.mat'
    pt_file = os.path.join(pt_folder, f'syn_data_300_GCN.pt')

    # 从.pt文件加载数据
    X = torch.load(pt_file)
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()

    print('Loaded data with shape:', X.shape)

    # 确保 X 为 float32 类型
    # 将X转换为float32类型并计算X*X^T
    X = X.astype(np.float32)

    # X = np.dot(X, X.T)
    X = compute_tensor_T2(X)
    
    # 使用归一化参数调用可视化函数
    visualize_emb(X, save_path=os.path.join(save_dir, f'{dataset}_GCN_sim.png'), normalize=True)

if __name__ == '__main__':
    main()







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