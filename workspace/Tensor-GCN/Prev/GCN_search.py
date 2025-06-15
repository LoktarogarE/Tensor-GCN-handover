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
    # L3 = get_L3(features, feat_cog, undecomposable).T # MTGCN take down the ".T"
    L3 = compute_T3(features, feat_cog, undecomposable).T

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
    'hidden_dim': [16, 32, 64, 128, 256, 512],  # 隐藏层维度 8, 
    'dropout': [0.4, 0.5, 0.6, 0.7],      # dropout比率 , 0.9 , 0.8
    'weight_decay': [5e-5, 5e-4, 5e-3, 5e-2]  # 权重衰减 , 5e-1
}

# 网格搜索最优参数
best_acc = 0
best_params = {}

# 创建结果存储字典，用于后续分析和可视化
results = {
    'hidden_dim': [],
    'dropout': [],
    'weight_decay': [],
    'accuracy': [],
    'auc': [],
    'f1': [],
    'recall': []
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
                    if epoch > 0:
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
                        
                        if(test_auc > 0.99):
                            torch.save(output, f'{dataset}_{epoch}_{hd}_{dp}_{wd}.pt')
                            # 可视化预测结果
                            # heatmap(output, y, test_mask)
                    # if train_loss > best_loss:
                    #     best_val_loss = train_loss
                    #     best_epoch = epoch
                    
                    # if epoch >= best_epoch + early_stop:
                    #     break

                model.eval()
                output = model(features, L, L3)
                
                f,auc,rec = all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, test_mask)  # 此处更改为test_mask
                fs.append(f)
                aucs.append(auc)
                recs.append(rec)

                test_acc = masked_accuracy(output, y_test, test_mask)
                test_accls.append(test_acc)
                print("{}layers, Final test  ACC: {} %".format(n_layer,np.round(test_acc * 100, 4)))
                
                # 存储当前参数组合的结果
                results['hidden_dim'].append(hd)
                results['dropout'].append(dp)
                results['weight_decay'].append(wd)
                results['accuracy'].append(test_acc)
                results['auc'].append(auc)
                results['f1'].append(f)
                results['recall'].append(rec)
                
                # 更新最佳参数
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_params = {'hidden_dim': hd, 'dropout': dp, 'weight_decay': wd, 'layer': n_layer}

    # torch.save(model.state_dict(), 'model_param.pkl')
    
#     if n_layers == 2:
#         plt.figure(figsize=(10, 8))
#         plt.title("Dataset: {}".format(dataset))
#         plt.plot(range(len(val_accls)), val_accls, label="validation", linewidth=3)
#         plt.plot(range(len(train_accls)), train_accls, label="train", linewidth=3)
#         plt.xlabel("n epochs")
#         plt.ylabel("Accuracy")
#         plt.legend()
#         plt.savefig('cora|train'+'.jpg')
#         # plt.show()


end_time = time.time();
print("运行时间：", end_time - start_time, " sec\n")

for i in range (begin_layer, n_layers+1):
    ind = i-begin_layer
    print("Layer {}, Acc:{}, F:{}, Auc:{}, Rec:{}".format(i, test_accls[ind], np.round(fs[ind], 4), np.round(aucs[ind],4),np.round(recs[ind], 4)))

average_acc = np.round(sum(test_accls)/len(test_accls) * 100, 4)
max_acc = np.round(max(test_accls) * 100, 4)
max_acc_layer = test_accls.index(max(test_accls)) + 1

print("Average Acc of all layers: ", average_acc)
print("Max Acc: ", max_acc, "with layer",max_acc_layer)
print("Best parameters: ", best_params)

# 创建参数分析图表
plt.figure(figsize=(18, 12))

# 1. hidden_dim 分析
plt.subplot(2, 2, 1)
hd_df = pd.DataFrame({
    'hidden_dim': results['hidden_dim'],
    'accuracy': results['accuracy'],
    'dropout': results['dropout'],
    'weight_decay': results['weight_decay']
})

# 对每个hidden_dim值，计算平均准确率
hd_grouped = hd_df.groupby('hidden_dim')['accuracy'].mean().reset_index()
hd_grouped['accuracy'] = hd_grouped['accuracy'] * 100  # 转换为百分比

plt.plot(hd_grouped['hidden_dim'], hd_grouped['accuracy'], 'o-', linewidth=2, markersize=8)
plt.title('Hidden Dimension vs Accuracy', fontsize=14)
plt.xlabel('Hidden Dimension', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(PARAM_GRID['hidden_dim'])

# 2. dropout 分析
plt.subplot(2, 2, 2)
dp_df = pd.DataFrame({
    'dropout': results['dropout'],
    'accuracy': results['accuracy'],
    'hidden_dim': results['hidden_dim'],
    'weight_decay': results['weight_decay']
})

# 对每个dropout值，计算平均准确率
dp_grouped = dp_df.groupby('dropout')['accuracy'].mean().reset_index()
dp_grouped['accuracy'] = dp_grouped['accuracy'] * 100  # 转换为百分比

plt.plot(dp_grouped['dropout'], dp_grouped['accuracy'], 'o-', linewidth=2, markersize=8, color='green')
plt.title('Dropout Rate vs Accuracy', fontsize=14)
plt.xlabel('Dropout Rate', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(PARAM_GRID['dropout'])

# 3. weight_decay 分析
plt.subplot(2, 2, 3)
wd_df = pd.DataFrame({
    'weight_decay': results['weight_decay'],
    'accuracy': results['accuracy'],
    'hidden_dim': results['hidden_dim'],
    'dropout': results['dropout']
})

# 对每个weight_decay值，计算平均准确率
wd_grouped = wd_df.groupby('weight_decay')['accuracy'].mean().reset_index()
wd_grouped['accuracy'] = wd_grouped['accuracy'] * 100  # 转换为百分比

plt.plot(wd_grouped['weight_decay'], wd_grouped['accuracy'], 'o-', linewidth=2, markersize=8, color='red')
plt.title('Weight Decay vs Accuracy', fontsize=14)
plt.xlabel('Weight Decay', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xscale('log')  # 使用对数刻度更好地显示weight_decay
plt.xticks(PARAM_GRID['weight_decay'])

# 4. 热力图分析 (hidden_dim vs dropout)
plt.subplot(2, 2, 4)
heatmap_data = pd.DataFrame({
    'hidden_dim': results['hidden_dim'],
    'dropout': results['dropout'],
    'accuracy': results['accuracy']
})

# 创建透视表
pivot_table = heatmap_data.pivot_table(
    values='accuracy', 
    index='dropout',
    columns='hidden_dim',
    aggfunc='mean'
) * 100  # 转换为百分比

# 绘制热力图
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5)
plt.title('Accuracy (%) by Hidden Dim and Dropout', fontsize=14)
plt.xlabel('Hidden Dimension', fontsize=12)
plt.ylabel('Dropout Rate', fontsize=12)

# 调整布局并保存
plt.tight_layout()
t = datetime.now()
str_t = t.strftime('%m.%d%H:%M')
plt.savefig(f'param_analysis_{dataset}_{str_t}.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存参数分析结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv(f'param_analysis_{dataset}_{str_t}.csv', index=False)

# 原始图表代码（注释掉）
# plt.figure(figsize=(10, 8))
# plt.title("Dataset: {}".format(dataset))
# plt.plot(range(1, len(test_accls) + 1), test_accls, linewidth=3)
# plt.xlabel("number of layers")
# plt.ylabel("Test Accuracy")
# plt.legend()
# # plt.show()
# t = datetime.now()
# str_t = t.strftime('%m.%d%H:%M')
# # plt.savefig('pic_res/{}/{} hd:{}|均:{}|max:{}({})'.format(dataset, str_t, hidden_dim, average_acc, max_acc, max_acc_layer)+'.jpg')
# plt.savefig('pic_res/{}/{}|max:{}({})|dp:{}wd:{}'.format(dataset, str_t, max_acc, max_acc_layer, dropout, weight_decay) + sub_message +'.jpg')
# with open('data.txt', 'a') as f:
#     for i in range (begin_layer, n_layers+1):
#         ind = i - begin_layer
#         f.write("Layer {}, Acc:{}, F:{}, Auc:{}, Rec:{}".format(i, test_accls[ind], np.round(fs[ind], 4), np.round(aucs[ind],4),np.round(recs[ind], 4))+'\n')


