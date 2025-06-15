#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#------------------
#   全局设定
#------------------
tl.set_backend('pytorch')
device = torch.device('cpu')  # device gcn全部为cpu
## 指令
# nohup python run_GCN_search.py > log/usps8:2.log &

dataset = "lung"
n_epochs = 200
learning_rate = 0.01   #prev 0.01 
sub_message = ""  # 表示保存时需要额外注明的信息
is_load_dict = False
state_dict = 'model_param.pkl'

hd_para = [8,16,32,64,128,256,512]  # 7
dp_para = [0.4,0.5,0.6,0.7,0.8,0.9] # 6
wd_para = [5e-4, 5e-3, 5e-2, 5e-1] # 3

# n_features, n_classes
for i in range(1):
    weight_times = 1
    if dataset == "colon" or dataset == "Colon":
        n_features, n_classes = [2000*weight_times, 2]
        dataset = "Colon"
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


print("dataset: {} | {} classes | {} features | deivce: {}".format(dataset,n_classes,n_features,device))

def l2_reg(model, weight_decay):
    reg = 0.0
    if weight_decay == 0:
        return reg
    for name, parameter in model.first_layer.named_parameters():
        if 'weight' in name:
            reg += weight_decay * (parameter ** 2).sum()
    return reg

features, y_train, y_test, train_mask, test_mask, adj = load_data(dataset)
y = y_train + y_test  # All y is y_train + y_test





for hidden_dim in hd_para:
    for dropout in dp_para:
        for weight_decay in wd_para:
            test_accls = []
            fs = []
            aucs = []
            recs = []
            for n_layer in range(1, 9):  # 1 ~ 8层

                print("hd:{}, dp:{}, wd:{}, layer {} ".format(hidden_dim, dropout, weight_decay, n_layer))

                model = GCN(n_layers=n_layer, n_features=n_features, hidden_dim=hidden_dim, dropout=dropout, n_classes=n_classes)
                if is_load_dict == True:
                    model.load_state_dict(torch.load(state_dict))
                loss_func = nn.CrossEntropyLoss(reduction='none')
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                train_accls = []
                for epoch in range(n_epochs):
                    # train
                    model.to(device)
                    model.train()
                    y_train = y_train.to(device)
                    y_test = y_test.to(device)
                    optimizer.zero_grad()
                    output = model(features, adj)
                    loss = masked_softmax_cross_entropy(loss_func, output, y_train, train_mask)
                
                    reg = l2_reg(model, weight_decay).to(device)
                    loss += (reg / len(train_mask))
                    # 获取loss作为tensor的值
                    train_loss = loss.item()
                    loss.backward()   # 计算gradient
                    optimizer.step()  # 更新parameter
                    if epoch % 10 == 0:
                        model.eval()
                        train_acc = masked_accuracy(output, y_train, train_mask)
                        test_acc = masked_accuracy(output, y_test, test_mask)
                        print("epoch {} | train ACC {} % | test ACC {} % | train loss {}".format(
                            epoch + 1,
                            np.round(train_acc * 100, 4),
                            np.round(test_acc * 100, 4),
                            np.round(train_loss, 4),
                        ))
                # test
                model.eval()
                output = model(features, adj)
                f,auc,rec = all_score(output.argmax(1), nn.functional.softmax(output, dim = 1), y, train_mask)
                fs.append(np.round(f,4))
                aucs.append(np.round(auc,4))
                recs.append(np.round(rec,4))

                test_acc = masked_accuracy(output, y_test, test_mask)
                test_accls.append(test_acc)
                print("{}layers, Final test  ACC: {} %".format(n_layer,np.round(test_acc * 100, 4)))
                    # torch.save(model.state_dict(), 'model_param.pkl')

            for i in range (0, 8):
                print("Layer {}, Acc:{}, F:{}, Auc:{}, Rec:{}".format(i+1, test_accls[i], np.round(fs[i], 4), np.round(aucs[i],4),np.round(recs[i], 4)))
            max_acc = np.round(max(test_accls) * 100, 6)
            max_ind = test_accls.index(max(test_accls)) 
            # print("Max Acc: ", max_acc, "in layer",max_ind+1)
            print("Layer {}, max_Acc:{}, F:{}, Auc:{}, Rec:{}".format(max_ind+1, np.round(test_accls[max_ind]*100,6), fs[max_ind], aucs[max_ind], recs[max_ind]))
            with open('result_k=5/{}_lr={}_{}.txt'.format(dataset, learning_rate, sub_message), 'a') as f:
                f.write("hd:{}|dp:{}|wd:{} || ".format(hidden_dim, dropout, weight_decay))
                f.write("Layer {}, max_Acc:{}, F:{}, Auc:{}, Rec:{}".format(max_ind+1, np.round(test_accls[max_ind]*100,6), fs[max_ind], aucs[max_ind], recs[max_ind]))
                f.write('\n')