#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tool import device
# from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix



def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
    loss = org_loss_func(preds, labels)
    _mask = torch.FloatTensor(mask).to(device)
    _mask /= _mask.sum()
    loss *= _mask
    return loss.mean()


def masked_accuracy(preds, labels, mask):
    acc = torch.eq(preds.argmax(1), labels).float()
    _mask = torch.FloatTensor(mask).to(device)
    _mask /= _mask.sum()
    acc *= _mask
    return acc.sum().item()


def all_score(y_pred, y_pred_softmax, y, mask):
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred[mask]   # only test samples counts in all_score computation
    y_pred_softmax = y_pred_softmax.detach().cpu().numpy()[mask]
    y = y.numpy()[mask]
    # Compute F-score
    f_score = f1_score(y, y_pred, average='weighted')

    cm = confusion_matrix(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=1) + FP)
    spec = np.mean(TN / (TN + FP))
    # Compute AUC

    if len(np.unique(y))==2:
        auc = roc_auc_score(y, y_pred)
    else:
        auc = roc_auc_score(y, y_pred_softmax,  multi_class='ovr')
    # auc = roc_auc_score(y, y_pred_softmax,  multi_class='ovr') 
    # auc = roc_auc_score(y, y_pred) # gli85 
    # Compute recall rate
    recall = recall_score(y, y_pred, average='micro')
    # print(f'F-score: {f_score:.4f}')
    # print(f'Purity: {purity:.4f}')
    # print(f'AUC: {auc:.4f}')
    # print(f'Recall rate: {recall:.4f}')
    return recall, f_score, auc, spec, prec






# import torch
# from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score
# import numpy as np

# def masked_softmax_cross_entropy(org_loss_func, preds, labels, mask):
#     loss = org_loss_func(preds, labels)
#     _mask = torch.FloatTensor(mask)
#     # 标准化
#     _mask /= _mask.sum()
#     loss *= _mask 
#     # 求平均
#     return loss.mean()


# def masked_accuracy(preds, labels, mask):
#     acc = torch.eq(preds.argmax(1), labels).float()
#     _mask = torch.FloatTensor(mask)
#     _mask /= _mask.sum()
#     acc *= _mask
#     return acc.sum().item()

# def all_score(y_pred, y_pred_softmax, y, mask):
#     y_pred = y_pred.detach().cpu().numpy()
#     y_pred_softmax = y_pred_softmax.detach().cpu().numpy()
#     y = y.numpy()
#     # Compute F-score
#     f_score = f1_score(y, y_pred, average='weighted')
#     # Compute purity (accuracy)
#     purity = accuracy_score(y, y_pred)
#     # Compute AUC

#     # auc = roc_auc_score(y, y_pred_softmax, multi_class='ovr')
#     auc = roc_auc_score(y, y_pred)  # 2 classes
#     # Compute recall rate
#     recall = recall_score(y, y_pred, average='weighted')
#     # print(f'F-score: {f_score:.4f}')
#     # print(f'Purity: {purity:.4f}')
#     # print(f'AUC: {auc:.4f}')
#     # print(f'Recall rate: {recall:.4f}')
#     return f_score, purity, auc, recall


# def visualize_tsne(all_input, all_output_labeled, epoch,mask,dataset,n_layer):
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#     import os
#     all_input=all_input[mask]
#     # all_output_labeled=all_output_labeled[mask]
#     all_output_labeled = all_output_labeled.cpu().detach().numpy() 
#     # all_output_labeled=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
#     #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2,
#     #     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#     #     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
#     all_output_labeled=all_output_labeled[mask]

#     # Perform t-SNE   early_exaggeration=36
#     tsne = TSNE(n_components=2, random_state=0, early_exaggeration=36)
#     X_test_2d = tsne.fit_transform(all_input)

#     # Plot t-SNE visualization
#     plt.figure(figsize=(6, 5))

#     # Plot data points colored by class]
#     colors = ['#0082807F', '#EE4C97FF','#FFDC91FF', '#008B457F', '#E18727FF']
#     # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'darkorange', 'limegreen', 'dodgerblue']
#     for i in range(len(np.unique(all_output_labeled))):
#         plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('%d' % i))
#         # plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('class%d' % i))
    
#     plt.xlabel(dataset)
#     plt.ylabel('tSNE')
#     # Add legend
#     # plt.legend()
#     # plt.legend(loc="upper right")


#     # Save plot
#     save_dir = f'./fig/tsne/{dataset}/'
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f'{dataset}_layer_{n_layer}_epoch_{epoch}.png'), dpi=750, bbox_inches='tight')





def visualize_tsne(all_input, all_output_labeled, epoch,mask,dataset,n_layer):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import os
    # layer_name = 'my_layer'
    print(all_output_labeled.shape)
    all_input=all_input.cpu()
    all_input=all_input[mask]
    all_output_labeled=all_output_labeled[mask]
    all_output_labeled = all_output_labeled.cpu().numpy() 

    # Perform t-SNE   early_exaggeration=36 random_state=0, ,  early_exaggeration=12
    tsne = TSNE(n_components=2, random_state=0)
    X_test_2d = tsne.fit_transform(all_input)

    # Plot t-SNE visualization
    plt.figure(figsize=(6, 5))

    # Plot data points colored by class]
    colors = ['#0082807F', '#EE4C97FF','#FFDC91FF', '#008B457F', '#E18727FF']
    # colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'darkorange', 'limegreen', 'dodgerblue']
    for i in range(len(np.unique(all_output_labeled))):
        plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('%d' % i))
        # plt.scatter(X_test_2d[all_output_labeled == i, 0], X_test_2d[all_output_labeled == i, 1], c=colors[i], label=('class%d' % i))
    
    plt.xlabel(dataset)
    plt.ylabel('tSNE')
    # Add legend
    # plt.legend()
    # plt.legend(loc="upper right")


    # Save plot
    save_dir = f'./fig/tsne_dec/{dataset}/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{dataset}_layer_{n_layer}_epoch_{epoch}.png'), dpi=750, bbox_inches='tight')

