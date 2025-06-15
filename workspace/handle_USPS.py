import numpy as np
import scipy.io as sio

# 设置随机种子，保证结果可复现（可选）
np.random.seed(42)

# 1. 加载 .mat 数据文件
# 假设文件名为 'input.mat'，且其中存有变量 'X' 和 'Y'
mat_contents = sio.loadmat('ZDataSet/USPS_1000.mat')
X = mat_contents['X']
Y = mat_contents['Y']

print("X 的尺寸:", X.shape)
print("Y 的尺寸:", Y.shape)

# 2. 定义抽样参数
num_classes = 10          # 总类别数为 10 类
samples_per_class = 100   # 每类有 100 个样本
select_per_class = 40     # 每类抽取 40 个样本

selected_indices = []

# 3. 从每个类别中按顺序抽取样本
for class_idx in range(num_classes):
    # 计算当前类别在数据集中的起始和结束索引
    start_idx = class_idx * samples_per_class
    end_idx = (class_idx + 1) * samples_per_class
    class_indices = np.arange(start_idx, end_idx)
    
    # 随机选择 select_per_class 个不重复的样本索引
    chosen_indices = np.random.choice(class_indices, select_per_class, replace=False)
    selected_indices.extend(chosen_indices)

# 转换为 numpy 数组，并排序（排序可以保持原始数据顺序，但不是必须的）
selected_indices = np.array(selected_indices)
selected_indices.sort()

# 4. 提取对应的 X 和 Y 样本
# 假设 X 的每一行代表一个样本，Y 同样按照样本顺序排列
X_sampled = X[selected_indices, :]
Y_sampled = Y[selected_indices, :]

print("抽样后 X 的尺寸:", X_sampled.shape)
print("抽样后 Y 的尺寸:", Y_sampled.shape)

# 5. 将抽样后的数据保存到新的 .mat 文件中，保持变量名 'X' 和 'Y'
sio.savemat('sampled_dataset.mat', {'X': X_sampled, 'Y': Y_sampled})

print("抽样后的数据已保存到 'sampled_dataset.mat'")


# import scipy.io
# from sklearn.model_selection import StratifiedShuffleSplit
# import numpy as np

# # 输入输出文件路径
# input_file = 'ZDataSet/USPS_1000.mat'
# output_file = 'ZDataSet/USPS_400.mat'

# # 加载原始数据
# try:
#     data = scipy.io.loadmat(input_file)
#     features = data['X']
#     labels = np.array([i//100 for i in range(1000)])  # 生成类别标签（前100类1，后续类推）
    
#     # 验证数据形状
#     assert features.shape[0] == 1000, "样本数量不符预期"
    
#     # 分层抽样（每类抽取40个样本）
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=400, train_size=600, random_state=42)
#     for _, indices in sss.split(features, labels):
#         sampled_features = features[indices]
#         sampled_labels = labels[indices]
    
#     # 保存新数据集
#     scipy.io.savemat(output_file, {
#         'X': sampled_features,
#         'Y': sampled_labels
#     })
    
#     print(f'成功抽取{sampled_features.shape[0]}个样本，保存至 {output_file}')
#     print('各类样本数量:', np.bincount(sampled_labels))

# except FileNotFoundError:
#     print(f"错误：输入文件 {input_file} 不存在")
# except KeyError:
#     print("错误：MAT文件中缺少 'data' 字段")
# except Exception as e:
#     print(f'处理失败: {str(e)}')

