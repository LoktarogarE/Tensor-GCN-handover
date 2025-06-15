# 目标：给数据集名字，返回所有数据(直接预处理好再拿进去)
import numpy as np
import torch
import pickle as pkl
from tool import *

# 实现变量保存，变量读取
def save_variable(v, filename):
    f = open(filename, 'wb')
    pkl.dump(v,f)
    f.close()
    return filename
def load_variable(filename):
    f = open(filename, 'rb')
    r = pkl.load(f)
    f.close()
    return r



## dataloader
def load_data(dataset_str):
    names = ['features', 'y_train', 'y_test', 'train_mask', 'test_mask', 'adj']
    objects = []
    for i in range(len(names)):
        objects.append(load_variable("Zdata/ind.{}.{}".format(dataset_str, names[i])))

    features, y_train, y_test, train_mask, test_mask, adj = tuple(objects)
    # 预处理
    features = preprocess_features(features).to(torch.float32)
    # features = torch.from_numpy(features).to(torch.float32)
    adj = preprocess_adj(adj).to(torch.float32)
    y_train = torch.FloatTensor(y_train).argmax(dim=1)
    y_test = torch.FloatTensor(y_test).argmax(dim=1)

    return features, y_train, y_test, train_mask, test_mask, adj

# def load_data_graph(dataset_str):
    



# 调试用
if __name__ == "__main__":
    features, y_train, y_test, train_mask, test_mask, adj = load_data("syn_data_100")
    # print(adj)
    print(features.shape)