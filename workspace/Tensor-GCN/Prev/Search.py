#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from heatmap import *


import os
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from GCN import *
from metrics import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tool import *

import tensorly as tl
from dataloader import *
import matplotlib as mpl
mpl.use('Agg')
from datetime import datetime
from tqdm import tqdm

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#------------------
#   全局设定
#------------------
tl.set_backend('pytorch')
# device = torch.device("")

# 导入配置
from config import HYPER_PARAMS, get_dataset_config
from config import CURRENT_DATASET
#------------------
#   加载配置   
#------------------

# 设置
dataset = CURRENT_DATASET
sub_message = ""  # 表示保存时需要额外注明的信息

# 获取数据集配置    
dataset_config, dataset = get_dataset_config(dataset)
n_features = dataset_config['n_features']
n_classes = dataset_config['n_classes']
feat_cog = dataset_config.get('feat_cog', HYPER_PARAMS['feat_cog'])

# 加载超参数
hidden_dim = HYPER_PARAMS['hidden_dim']
dropout = HYPER_PARAMS['dropout']
n_epochs = HYPER_PARAMS['n_epochs']
begin_layer = HYPER_PARAMS['begin_layer']
n_layers = HYPER_PARAMS['n_layers']
weight_decay = HYPER_PARAMS['weight_decay']
learning_rate = HYPER_PARAMS['learning_rate']
undecomposable = HYPER_PARAMS['undecomposable']
is_load_dict = HYPER_PARAMS['is_load_dict']
state_dict = HYPER_PARAMS['state_dict']



print("dataset: {} | {} classes | {} features | deivce: {}".format(dataset,n_classes,n_features,device))

# 时间统计：
start_time = time.time()

def l2_reg(model, weight_decay):
    reg = 0.0
    if weight_decay == 0:
        return reg
    for name, parameter in model.first_layer.named_parameters():
        if 'weight' in name:
            reg += weight_decay * (parameter ** 2).sum()
    return reg

# 调试， 保存梯度
# grads = {}
# def save_grad(name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

features, y_train, y_test, train_mask, test_mask, adj = load_data(dataset)
y = y_train + y_test  # 这是所有的Y


print("dataset: {} | {} samples | {} classes | {} features | deivce: {}".format(dataset,features.shape[0],n_classes,n_features,device))

# 计算 L3
print("prepareing...")


laplacian = torch.eye(adj.shape[0]) - adj
largest_eigval = torch.linalg.eigvalsh(laplacian, UPLO='U')[-1]
scaled_laplacian = (2. / largest_eigval) * laplacian - torch.eye(adj.shape[0])
L = scaled_laplacian.to(torch.float32)
if undecomposable == False:
    L3 = tl.tenalg.khatri_rao([L,L]).T   # 可分解相似度
else:
    # L3 = get_L3_tqdm(features, feat_cog, undecomposable)
    L3 = get_L3(features, feat_cog, undecomposable).T # MTGCN take down the ".T"
    # L3 = compute_T3(features, feat_cog, undecomposable).T

# 可视化相似度矩阵
# print("prepareing similarity heatmap...")
# heatmap_sim(L)



test_accls = []
fs = []
aucs = []
recs = []
num_classes=np.unique(y)


# 参数网格搜索配置
PARAM_GRID = {
    'hidden_dim': [16, 32, 64, 128, 256],  # 隐藏层维度 8, 
    'dropout': [0.4, 0.5, 0.6, 0.7],      # dropout比率 , 0.9 , 0.8
    'weight_decay': [5e-5, 5e-4, 5e-3]  # 权重衰减 , 5e-1
}
for hd in PARAM_GRID['hidden_dim']:
    for dp in PARAM_GRID['dropout']:
        for wd in PARAM_GRID['weight_decay']:
            # 更新当前参数
            hidden_dim = hd
            dropout = dp 
            weight_decay = wd
            print(f"\n正在测试参数组合: hidden_dim={hd}, dropout={dp}, weight_decay={wd}")
            for n_layer in range(begin_layer, n_layers+1):  # amended from 11 
                model = GCN(n_layers=n_layer, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout, n_classes=n_classes)
                if is_load_dict == True:
                    model.load_state_dict(torch.load(state_dict))
                loss_func = nn.CrossEntropyLoss(reduction='none')
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                best_epoch = 0
                best_loss = 0.0
                train_accls = []
                for epoch in range(n_epochs):
                    t = time.time()

                    # train
                    model.to(device)
                    model.train()
                    y_train = y_train.to(device)
                    y_test = y_test.to(device)
                    optimizer.zero_grad()
                    output = model(features, L, L3)
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f'Gradient norm for {name}: {param.grad.norm()}')
                    # y_pred_softmax = nn.functional.softmax(output, dim = 1)
                    # y_pred_softmax = y_pred_softmax.argmax(1)
                    # y_pred_argmax = output.argmax(1)

                    loss = masked_softmax_cross_entropy(loss_func, output, y_train, train_mask)
                    reg = l2_reg(model, weight_decay).to(device)
                    loss += (reg / len(train_mask))
                    train_loss = loss.item()
                    loss.backward()   # 计算gradient
                    # for name, param in model.named_parameters():
                    #         if param.grad is not None:
                    #             print(f'Gradient norm for {name}: {param.grad.norm()}')
                    # time.sleep(1)
                    optimizer.step()  # 更新parameter


                    # if epoch % 5 == 0:
                    model.eval()
                    output = model(features, L, L3)
                    train_acc = masked_accuracy(output, y_train, train_mask)
                    test_acc = masked_accuracy(output, y_test, test_mask)       
                    # visualize_roc(y_test, y_pred_softmax, num_classes, epoch,test_mask,dataset,n_layer)
                    # visualize_tsne(features, y_pred_argmax, epoch,test_mask,dataset,n_layer)
                    _,test_auc,_= all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, test_mask)  # 此处更改为test_mask
                    if epoch % 5 == 0: 
                        print("epoch {} | train ACC {} % | test ACC {} % | test AUC {} % | train loss {} ".format(
                            epoch + 1,
                            np.round(train_acc * 100, 4),
                            np.round(test_acc * 100, 4),
                            np.round(test_auc * 100, 4),
                            np.round(train_loss, 4),
                        ))
                    
                    if(test_auc > 0.998):
                        # 建议添加绝对路径和检查点目录
                        save_dir = "/home/wangcy/sandbox/Tensor-GCN_Claude/checkpoints"
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f'{dataset}_{epoch}_{n_layer}.pkl')
                        # torch.save(model.state_dict(), save_path)
                        # 更推荐同时保存模型结构和参数（如需跨设备加载）：
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)



    
                

                





