# 
# ------------------------------------------
# Dataset      Instances  Features  Classes  Rate

# ALLAML       72         7129      2        47 + 25 | 115.7
# Leukemia     72         7070      2        25 + 47 | 175.3
# GLI_85       85         22283     2        26 + 59 | 10.6  
# Prostate_GE  102        5966      2        50 + 52 | 29.13 
# Lung         203        3312      5        139 17 21 20 6| 23.3
# GCM  

                                             
# GLIOMA 50 samples, 4434 features 4 classes

# CLA_BRA      180        49151     4     # 暂无mat
# Colon        62         2000      2        22 + 40 | 92.2 # 
# CLL_SUB      111        11340     3        11 + 49 + 51 | 109 # 
# arcene       200        10000     2        88 + 112 | 3873695 #
# SMK_CAN      187        19993     2        90 + 97 | 77.2 # 
# nci9  # 效果逆天差
# lymphoma     96         4026      9        [46. 10.  9. 11.  6.  6.  4.  2.  2.] # 
# COIL20 1440 samples, 3 features, 20 classes  ??? I don't understand
# USPS_1000    1000       256       10      11 # 太大
# ------------------------------------------


from scipy.io import loadmat
import numpy as np
import networkx as nx
import pickle
import time
import torch
from torch import nn
import math

def add_salt_pepper_noise(data, noise_ratio=0.1, salt_vs_pepper=0.5):
    """
    为数据集添加 Salt & Pepper 噪声
    
    参数:
        data: numpy数组，输入数据
        noise_ratio: float, 噪声比例，默认为0.1（10%的像素会被噪声影响）
        salt_vs_pepper: float, salt噪声（1）与pepper噪声（0）的比例，默认为0.5（各占一半）
    
    返回:
        添加了噪声的数据副本
    """
    # 创建数据的副本，避免修改原始数据
    noisy_data = np.copy(data)
    
    # 计算要添加噪声的元素总数
    total_elements = data.size
    num_noise_elements = int(total_elements * noise_ratio)
    
    # 随机选择要添加噪声的元素位置
    noise_indices = np.random.choice(total_elements, num_noise_elements, replace=False)
    
    # 将选中的位置展平为一维索引
    flat_noisy_data = noisy_data.flatten()
    
    # 根据salt_vs_pepper比例决定salt（1）和pepper（0）噪声的数量
    num_salt = int(num_noise_elements * salt_vs_pepper)
    
    # 使用数据的实际范围而不是数据类型的极限值：
    # 计算数据的最大值和最小值
    max_value = np.max(data)
    min_value = np.min(data)
    
    # 如果数据范围太小，使用更合理的默认值
    if max_value == min_value:
        if np.issubdtype(data.dtype, np.floating):
            max_value = 1.0
            min_value = 0.0
        else:
            max_value = 255 if max_value > 0 else 1
            min_value = 0
    
    # 添加salt噪声（设为数据的最大值）
    salt_indices = noise_indices[:num_salt]
    flat_noisy_data[salt_indices] = max_value
    
    # 添加pepper噪声（设为数据的最小值）
    pepper_indices = noise_indices[num_salt:]
    flat_noisy_data[pepper_indices] = min_value
    
    # 将展平的数组重新调整为原始形状
    noisy_data = flat_noisy_data.reshape(data.shape)
    
    return noisy_data

def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


k = 4  # K+1 neighbors
dataset_list = ["Colon", "Leukemia", "ALLAML", "SMK_CAN", "GLI_85", "Prostate_GE", "CLL_SUB","Lung"]
# "arcene", "lymphoma", "nci9", 
dataset_list = ["Lung"]
introduce_noise = False # 是否添加椒盐噪声，默认为 False

for dataset_str in dataset_list:
    # Initialize
    n_classes = 2
    para = np.asarray([1,1,1])
    X, Y = 'X', 'Y'
    is_syn = False  # 标记是否为合成数据集
    # para是数据中每一类 在训练集中占据的数量，根据这个数量来分开数据集
    # if dataset_str == "GCM":
    if dataset_str == "GMM_data_300":
        n_classes = 3
        para = np.asarray([20,20,20])  # 按照8:2的比例划分训练集和测试集
        is_syn = True  # 设置为合成数据集
    elif dataset_str == "GMM_data_600":
        n_classes = 3
        para = np.asarray([40,40,40])  # 按照8:2的比例划分训练集和测试集
        is_syn = True  # 设置为合成数据集
    elif dataset_str == "syn_data_100":
        n_classes = 3
        para = np.asarray([7,7,7])  # 按照8:2的比例划分训练集和测试集
        is_syn = True  # 设置为合成数据集
    elif dataset_str == "syn_data_300":
        n_classes = 3
        para = np.asarray([20,20,20])  # 按照8:2的比例划分训练集和测试集
        is_syn = True  # 设置为合成数据集  
    if dataset_str == "TOX_171":
        para = np.asarray([9,9,8,9])
        n_classes = 4
    if dataset_str == "nci9":
        para = np.asarray([3,3,2,1,2,2,2,2,1])
        n_classes = 9
    if dataset_str == "lymphoma":
        para = np.asarray([9,2,2,2,1,1,1,1,1])
        n_classes = 9
    if dataset_str == "acrene":
        para = np.asarray([17, 22])
    if dataset_str == "Colon":
        n_classes = 2
        para = np.asarray([4, 8])
    elif dataset_str == "Leukemia":
        n_classes = 2
        para = np.asarray([5, 9])
    elif dataset_str == "ALLAML":
        n_classes = 2
        para = np.asarray([9, 5])
    elif dataset_str == "SMK_CAN":
        n_classes = 2
        para = np.asarray([18, 19])
    elif dataset_str == "GLI_85":
        n_classes = 2
        # para = np.asarray([3,7]) #GLI85 即：第一类10个，第二类10个
        para = np.asarray([6,11]) # GLI85 8:2 
    elif dataset_str == "Prostate_GE":
        n_classes = 2
        para = np.asarray([10, 11])
    elif dataset_str == "CLL_SUB":
        n_classes = 3
        para = np.asarray([2, 9, 10])
        similarity_mode = 10000
    elif dataset_str == "CLA_BRA":
        n_classes = 4
    elif dataset_str == "Lung":
        n_classes = 5
        # para = np.asarray([13,2,2,2,1])  # Lung
        para = np.asarray([28,3,4,4,1])  # 8:2 划分
    elif dataset_str == "GLIOMA":
        n_classes = 4
        para = np.asarray([3,1,3,3]) #GLIOMA  样本数为50， 因此只取10个样本 8:2 无问题
        similarity_mode = 1
    elif dataset_str == "USPS_1000":
        n_classes = 10
        # para = np.asarray([8,8,8,8,8,8,8,8,8,8]) #USPS_1000
        para  = np.asarray([20,20,20,20,20,20,20,20,20,20])  # 2:8  

    annots = loadmat("ZDataSet/{}.mat".format(dataset_str))
    # annots = loadmat("{}.mat".format(dataset_str))
    if is_syn:  # 如果是合成数据集，使用'data'和'gt'字段
        X = annots['data']
        Y = annots['gt']
    else:  # 否则使用'X'和'Y'字段
        X = annots['{}'.format(X)]
        Y = annots['{}'.format(Y)]
    # n_samples, n_classes
    N = len(X)

    # 加入encoder的实验
    # feature_in = X[1].shape
    # print(type(feature_in))
    # feature_out = (math.floor(feature_in[0]/3),)
    # print(feature_out)
    # X = torch.from_numpy(X).float()
    # # encoder = nn.Linear(in_features=feature_in, out_features=feature_out)
    # # encoder = nn.Linear(22283, 7427)
    # encoder = nn.Sequential(
    # nn.Linear(22283, 7427),
    # nn.ReLU()
    # )
    # X = encoder(X)
    # X = X.detach().numpy()

    # 中和 记录为 +1 和 -1 的数据集
    if dataset_str == "Colon" or "Leukemia":
        for item in Y:
            if item < 0:
                item += 3
    # 预统计，获取samples，features
    # print(annots)
    # print(X[0].shape)

    #将原本的Y转化为one-hot格式， Y_onehot shape: (n_samples, n_classes), 并计算每种标签的数量
    count = np.zeros((n_classes))
    Y_onehot = np.zeros((N, n_classes))
    i = 0
    for item in Y:
        Y_onehot[i][item-1] = 1
        # print(i, Y[i])
        count[item-1] += 1
        i += 1
    train_mask = np.full(N, False, dtype=bool)
    for i in range(N):
        item = Y[i]
        if (para[item-1] > 0):
            train_mask[i] = 1
            para[item-1] -= 1


    Y = Y_onehot
    test_mask = ~train_mask  # train_mask 直接取反，新的代码中去掉了val
    y_train = np.zeros(Y.shape)
    y_test = np.zeros(Y.shape)
    y_train[train_mask, :] = Y[train_mask, :]
    y_test[test_mask, :] = Y[test_mask, :]

    if introduce_noise == True:
        noisy_X = add_salt_pepper_noise(X, noise_ratio=0.3, salt_vs_pepper=0.5)
        X = noisy_X.astype(X.dtype)
    # 获取图结构和adjacency matrix
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1, N):
            S[i][j] = np.linalg.norm(X[i]-X[j])
            S[j][i] = S[i][j]
    # 求S的中位数为sigma, 高斯核函数作为相似度
    S_vec = S.reshape(1, N*N)
    # sigma = np.median(S_vec)
    sigma = np.mean(S_vec)
    print("Sigma", sigma)

    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N)) #编号
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0]) #对距离进行排序
        b = dist_with_index
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
        #构建邻接矩阵
        # neighbours_id [id1, id2, id3, ...]
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]**2 / (2 * sigma**2))
            A[j][i] = A[i][j]

    # 调试输出
    # print(type(A), A.shape)
    # print("adj-I", (A-np.eye(A.shape[0])).sum())
    print("Similarity average: ", A.mean())
    # print(A)
    print("Class Count:", count)    
    # print("X", annots['X'].shape, X, " Y:", annots['Y'].shape, Y)
    # for i in range(N):
    #     print(i, y_train[i], train_mask[i])

    # GLI = loadmat("GLI_85_affinity_mat.mat")
    # adj_p = GLI['P_hat']


    save_variable(X, "Zdata/ind.{}.{}".format(dataset_str, "features"))
    save_variable(y_train, "Zdata/ind.{}.{}".format(dataset_str, "y_train"))
    save_variable(y_test, "Zdata/ind.{}.{}".format(dataset_str, "y_test"))
    save_variable(train_mask, "Zdata/ind.{}.{}".format(dataset_str, "train_mask"))
    save_variable(test_mask, "Zdata/ind.{}.{}".format(dataset_str, "test_mask"))
    save_variable(A, "Zdata/ind.{}.{}".format(dataset_str, "adj"))   # knn 构图
    # # # save_variable(adj_p, "Zdata/ind.{}.{}".format(dataset_str, "adj"))  # 采用最优构图
    print(dataset_str, "| Parameters Saved!")

print("Complete!")


