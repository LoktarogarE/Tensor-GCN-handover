#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import os
from datetime import datetime
import numpy as np
from tool import device
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, confusion_matrix



# from config import HYPER_PARAMS, get_dataset_config
from config import CURRENT_DATASET
dataset = CURRENT_DATASET

def heatmap_sim(L3):
    # 确保输入是numpy数组
    L3 = L3.detach().cpu().numpy() if torch.is_tensor(L3) else L3
    # 关闭交互式显示模式
    plt.ioff()
    # 绘制热力图 RdBu_r
    plt.figure(figsize=(10, 8))
    sns.heatmap(L3, annot=False, fmt='.2f', annot_kws={'size':8},
                cmap='viridis', cbar=True, 
                vmin=np.percentile(L3, 5), vmax=np.percentile(L3, 95),  # 使用百分位数作为显示范围
                cbar_kws={'label': 'Similarity Score'})
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Node Index', fontsize=12)
    plt.title('Similarity Matrix Heatmap', fontsize=14, pad=20)
    
    # 创建保存目录
    save_dir = './visual_pig/'
    os.makedirs(save_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%m%d_%H%M')
    plt.savefig(os.path.join(save_dir, f'sim_{dataset}_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def heatmap(y_pred, y, mask):
    
    # y_pred = y_pred.detach().cpu().numpy()
    # y_pred = y_pred[mask]   # only test samples counts in
    # y = y.detach().cpu().numpy()[mask]
    # cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    y_pred = y_pred.detach().cpu().numpy()
    print(y_pred.shape)
    # 计算y_pred的转置与自身的矩阵乘积
    y_pred = np.matmul(y_pred.T, y_pred)
    sns.heatmap(y_pred, annot=True, cmap='Reds')


    plt.xlabel('pred')
    plt.ylabel('label')
    plt.title('confusion matrix heatmap')
    plt.legend()
    plt.savefig('lung|heatmap' + '.jpg')




def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
    loss = org_loss_func(preds, labels)
    _mask = torch.FloatTensor(mask).to(device)
    _mask /= _mask.sum()
    loss *= _mask
    return loss.mean()


def masked_accuracy(preds, labels, mask):    # 确保所有张量在同一设备上
    preds = preds.to(device)
    labels = labels.to(device)
    acc = torch.eq(preds.argmax(1), labels).float()
    _mask = torch.FloatTensor(mask).to(device)
    _mask /= _mask.sum()
    acc *= _mask
    return acc.sum().item()

# def masked_accuracy(preds, labels, mask):  
#     _mask = torch.from_numpy(mask)
#     # labels = labels[_mask]
#     # preds = preds[_mask]
#     acc = torch.eq(preds.argmax(1)[_mask], labels[_mask]).float()
#     return acc.sum().item()




def all_score(y_pred, y_pred_softmax, y, mask):
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred[mask]   # only test samples counts in all_score computation
    y_pred_softmax = y_pred_softmax.detach().cpu().numpy()[mask]
    y = y.numpy()[mask]
    f_score = f1_score(y, y_pred, average='weighted')
    if len(np.unique(y))==2:
        auc = roc_auc_score(y, y_pred)
    else:
        auc = roc_auc_score(y, y_pred_softmax,  multi_class='ovr')
    recall = recall_score(y, y_pred, average='weighted')
    return f_score, auc, recall




def visualize_roc(y_true, y_scores, num_classes, epoch,mask,dataset,n_layer):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    from scipy import interp

    y_scores=y_scores.detach().cpu()[mask]
    y_true=y_true.detach().cpu()[mask]


    # Perform one-hot encoding if not already done
    if len(y_true.shape) == 1:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(categories=[range(num_classes)])
        y_true_bin = encoder.fit_transform(y_true.reshape(-1, 1)).toarray()
    else:
        y_true_bin = y_true

    # Plot ROC curves
    plt.figure()
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()

    for j in range(y_scores.shape[1]):
        if len(np.unique(y_true_bin[:, j])) > 1:
            fpr_dict[j], tpr_dict[j], _ = roc_curve(y_true_bin[:, j], y_scores[:, j])
            roc_auc_dict[j] = auc(fpr_dict[j], tpr_dict[j])
            plt.plot(fpr_dict[j], tpr_dict[j], label=f'ROC curve of class {j} (area = {roc_auc_dict[j]:.2f})')

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, label=f'micro-average ROC curve (area = {roc_auc_micro:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in fpr_dict.keys()]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(y_scores.shape[1]):
        if i in fpr_dict.keys():
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

    mean_tpr /= y_scores.shape[1]
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    plt.plot(fpr_macro, tpr_macro, label=f'macro-average ROC curve (area = {roc_auc_macro:.2f})',
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 保存图形
    save_dir = f'./roc/{dataset}/'  # 使用参数中的数据集名称
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset}_layer_{n_layer}_epoch_{epoch}.png'), dpi=750, bbox_inches='tight')



def visualize_tsne(all_input, all_output_labeled, epoch,mask,dataset,n_layer):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import os
    all_output_labeled=all_output_labeled[mask]

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_test_2d = tsne.fit_transform(all_input)

    # Plot t-SNE visualization
    plt.figure(figsize=(6, 5))

    # Plot data points colored by class
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'darkorange', 'limegreen', 'dodgerblue']
    for i in range(len(np.unique(all_output_labeled))):
        plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('%d' % i))
        # plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('class%d' % i))
    
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    # Add legend
    # plt.legend()
    plt.legend(loc="upper right")


    # Save plot
    save_dir = f'./fig/tsne/{dataset}/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset}_layer_{n_layer}_epoch_{epoch}.png'), dpi=750, bbox_inches='tight')