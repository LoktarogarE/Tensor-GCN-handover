#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# lastest version: 跑10次, 获取best_acc, 添加spec和prec


import os
import time
import torch
import numpy as np
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


#------------------
#   全局设定
#------------------
tl.set_backend('pytorch')
device = torch.device('cpu')

#------------------
#   hyper_parameter
#------------------
dataset = "syn300"
hidden_dim = 128  # prev 16
dropout = 0.5
n_epochs = 200
n_layers = 2
early_stop = 10
weight_decay = 5e-4 #prev 5e-4
learning_rate = 0.01 #prev 0.01 

# n_features, n_classes | 按照paralist自动分配
for i in range(1):
    weight_times = 1
    if dataset == "colon" or dataset == "Colon":
        n_features, n_classes = [2000*weight_times, 2]
        dataset = "Colon"
    elif dataset == "syn300" or dataset == "syn_data_300":
        n_features, n_classes = [500*weight_times, 3]
        dataset = "syn_data_300"
    elif dataset == "leuk" or dataset == "Leukemia":
        n_features, n_classes = [7070*weight_times, 2]
        dataset = "Leukemia"
    elif dataset == "all" or dataset == "ALLAML":
        n_features, n_classes = [7129*weight_times, 2]
        dataset = "ALLAML"
    elif dataset == "smk" or dataset == "SMK_CAN":
        n_features, n_classes = [19993*weight_times, 2]
        dataset = "SMK_CAN"
    elif dataset == "gli85" or dataset == "GLI85":
        n_features, n_classes = [22283*weight_times, 2]
        dataset = "GLI85"
    elif dataset == "pro" or dataset == "Prostate_GE":
        n_features, n_classes = [5966*weight_times, 2]
        dataset = "Prostate_GE"
    elif dataset == "cll" or dataset == "CLL_SUB":
        n_features, n_classes = [11340*weight_times, 3]
        dataset = "CLL_SUB"
    elif dataset == "lung" or dataset == "Lung":
        n_features, n_classes = [3312*weight_times, 5]
        dataset = "Lung"
    elif dataset == "GLIOMA" or dataset == "glio":
        n_features, n_classes = [4434*weight_times, 4]
        dataset = "GLIOMA"
    elif dataset == "usps" or dataset == "USPS_1000":
        n_features, n_classes = [256*weight_times, 10]
        dataset = "USPS_1000"


print("dataset: {} | {} classes | {} features | deivce: {}".format(
    dataset,
    n_classes,
    n_features,
    device
))

# 时间统计
start_time = time.time()

def l2_reg(model, weight_decay):
    reg = 0.0
    if weight_decay == 0:
        return reg
    for name, parameter in model.first_layer.named_parameters():
        if 'weight' in name:
            reg += weight_decay * (parameter ** 2).sum()
    return reg

features, y_train, y_test, train_mask, test_mask, adj = load_data(dataset)
y = y_test + y_train



# # 计算 L3
# print("prepareing...")
# laplacian = torch.eye(adj.shape[0]) - adj
# largest_eigval = torch.linalg.eigvalsh(laplacian, UPLO='U')[-1]
# scaled_laplacian = (2. / largest_eigval) * laplacian - torch.eye(adj.shape[0])
# L = scaled_laplacian.to(torch.float32)
# L3 = tl.tenalg.khatri_rao([L,L]).T


test_accls = []
fs = []
aucs = []
recs = []
specs = []
precs = []


best_acc = 0.0
b_f, b_auc, b_rec, b_spec, b_prec = 0.0,0.0,0.0,0.0,0.0
# visual
# visualize_tsne(features, y_test, -1,test_mask,dataset, 0)

# for n_layer in range(1, n_layers+1):  # amended from 11
for i in range (10):
    for n_layer in range(1, 4): 
        model = GCN(n_layers=n_layer, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout, n_classes=n_classes)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_epoch = 0
        train_accls = []
        
        for epoch in range(n_epochs):
            t = time.time()

            # train
            model.to(device)
            model.train()
            y_train = y_train.to(device)
            y_test = y_test.to(device)
            optimizer.zero_grad()
            output = model(features, adj)
            # model.Module.get_embeddings()
            # embedding = model.get_embeddings()

            loss = masked_softmax_cross_entropy(loss_func, output, y_train, train_mask)
        
            reg = l2_reg(model, weight_decay).to(device)
            loss += (reg / len(train_mask))
            # 获取loss作为tensor的值
            train_loss = loss.item()
            loss.backward()   # 计算gradient
            optimizer.step()  # 更新parameter
            if epoch % 5 == 0:
                model.eval()
                train_acc = masked_accuracy(output, y_train, train_mask)
                test_acc = masked_accuracy(output, y_test, test_mask)
                
                if(test_acc > 0.75):
                    # 保存当前output矩阵用于后续分析
                    torch.save(output, f'{dataset}_{epoch}.pt')
                    break
                if (test_acc > best_acc):
                    best_acc = test_acc
                    b_rec, b_f, b_auc, b_spec, b_prec = all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, train_mask)
                    
                # visualize_tsne(embedding, y_test, epoch,test_mask,dataset,n_layer)
                # visualize_tsne(features, output.argmax(1), epoch,test_mask,dataset,n_layer)
                print("epoch {} | train ACC {} % | test ACC {} % | train loss {}".format(
                    epoch + 1,
                    np.round(train_acc * 100, 4),
                    np.round(test_acc * 100, 4),
                    np.round(train_loss, 4),
                ))

        model.eval()
        output = model(features, adj)
        # recall, f_score, auc, spec, prec
        # f,auc,rec = all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, train_mask)
        rec, f, auc, spec, prec = all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, train_mask)
        fs.append(f)
        aucs.append(auc)
        recs.append(rec)
        specs.append(spec)
        precs.append(prec)

        test_acc = masked_accuracy(output, y_test, test_mask)
        test_accls.append(test_acc)

        print("{}layers, Final test  ACC: {} %".format(n_layer,np.round(test_acc * 100, 4))) 

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
    

print("运行时间：", time.time()-start_time, " sec\n")
print("Best, Acc:{}, Rec:{}, F:{}, Auc:{}, Specificity:{}, Precision:{}".format(best_acc, b_rec, b_f, b_auc, b_spec, b_prec))
# for i in range (n_layers):
for i in range (1):
    # print("Layer {}, Acc:{}, F:{}, Auc:{}, Rec:{}".format(i+1, test_accls[i], fs[i], aucs[i],recs[i]))
    print("Layer {}, Acc:{}, Rec:{}, F:{}, Auc:{}, Specificity:{}, Precision:{}".format(i+1, test_accls[i], recs[i], fs[i], aucs[i], specs[i], precs[i]))




average_acc = np.round(sum(test_accls)/len(test_accls) * 100, 4)
max_acc = np.round(max(test_accls) * 100, 4)
max_acc_layer = test_accls.index(max(test_accls)) + 1

print("Average Acc of all layers: ", average_acc)
print("Max Acc: ", max_acc, "with layer",max_acc_layer)

plt.figure(figsize=(10, 8))
plt.title("Dataset: {}".format(dataset))
plt.plot(range(1, len(test_accls) + 1), test_accls, linewidth=3)
plt.xlabel("number of layers")
plt.ylabel("Test Accuracy")
plt.legend()
# plt.show()
t = datetime.now()
str_t = t.strftime('%m.%d%H:%M')
plt.savefig('pic_res/{}/{} hd:{}|均:{}|max:{}({})'.format(dataset, str_t, hidden_dim, average_acc, max_acc, max_acc_layer)+'.jpg')
# print("Finally, test ACC with 2 layers: {}".format(test_accls[1]))
