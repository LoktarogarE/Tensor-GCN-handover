#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import tensorly as tl

# 设置后端
tl.set_backend('pytorch')

# 当前选择的数据集
CURRENT_DATASET = "Lung"

# 训练相关超参数
HYPER_PARAMS = {
    'hidden_dim': 256,
    'dropout': 0.4,
    'n_epochs': 200, #400
    'begin_layer': 3,  # 从多少层开始 默认1
    'n_layers':5,     # 决定最多跑到多少层
    'weight_decay': 5e-5,
    'learning_rate': 0.01,
    'weight_times': 3,
    'undecomposable': True,  # False => decomposable
    'feat_cog': 1e-9,  # ALL +0.5, leuk +0.1, usps 1e-5, other +1e-9
    'is_load_dict': False,
    'state_dict': 'model_param.pkl'
}

# 数据集配置
DATASET_CONFIGS = {
    'TOX_171': {'n_features': 5748, 'n_classes': 4, 'feat_cog': 1e-9},
    'nci9': {'n_features': 9712, 'n_classes': 9},
    'lymphoma': {'n_features': 4026, 'n_classes': 9},
    'arcene': {'n_features': 10000, 'n_classes': 2},
    'Colon': {'n_features': 2000, 'n_classes': 2},
    'Leukemia': {'n_features': 7070, 'n_classes': 2, 'feat_cog': 0.1},
    'ALLAML': {'n_features': 7129, 'n_classes': 2, 'feat_cog': 0.5},
    'SMK_CAN': {'n_features': 19993, 'n_classes': 2},
    'GLI85': {'n_features': 22283, 'n_classes': 2},
    'Prostate_GE': {'n_features': 5966, 'n_classes': 2},
    'CLL_SUB': {'n_features': 11340, 'n_classes': 3},
    'Lung': {'n_features': 3312, 'n_classes': 5},
    'GLIOMA': {'n_features': 4434, 'n_classes': 4},
    'USPS_1000': {'n_features': 256, 'n_classes': 10, 'feat_cog': 1e-9},
    'GMM_data_300': {'n_features': 500, 'n_classes': 3, 'feat_cog': 0.7},
    'GMM_data_600': {'n_features': 500, 'n_classes': 3, 'feat_cog': 0.2},
    'syn_data_100': {'n_features': 500, 'n_classes': 3, 'feat_cog': 1e-3},
    'syn_data_300': {'n_features': 500, 'n_classes': 3}
}

def get_dataset_config(dataset_name):
    """获取数据集配置
    Args:
        dataset_name: 数据集名称的小写或标准名称
    Returns:
        dict: 包含n_features、n_classes等配置的字典
    """
    # 标准化数据集名称
    name_mapping = {
        'tox': 'TOX_171',
        'nci': 'nci9',
        'lym': 'lymphoma',
        'arc': 'arcene',
        'colon': 'Colon',
        'leuk': 'Leukemia',
        'all': 'ALLAML',
        'smk': 'SMK_CAN',
        'gli85': 'GLI85',
        'pro': 'Prostate_GE',
        'cll': 'CLL_SUB',
        'lung': 'Lung',
        'glio': 'GLIOMA',
        'usps': 'USPS_1000',
        'gmm300': 'GMM_data_300',
        'gmm600': 'GMM_data_600',
        'syn100': 'syn_data_100',
        'syn300': 'syn_data_300'
    }
    
    std_name = name_mapping.get(dataset_name.lower(), dataset_name)
    config = DATASET_CONFIGS.get(std_name, {})
    
    if config:
        # 应用weight_times
        config = config.copy()
        config['n_features'] *= HYPER_PARAMS['weight_times']
        return config, std_name
    else:
        raise ValueError(f'未知的数据集名称: {dataset_name}')




