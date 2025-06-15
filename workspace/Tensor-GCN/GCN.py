#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from tool import get_support_MTGCN
from tool import device
from dataloader import *

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        

    # def forward(self, inputs, adj):
    #     # inputs: (N, n_channels), adj: sparse_matrix (N, N)
    #     inputs = self.dropout(inputs)  # H^(l)
    #     support = get_support(inputs, adj) 
    #     output = torch.mm(support.to(device), self.weight)
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output
    def forward(self, inputs, L, L3):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        inputs = self.dropout(inputs)  # H^(l)
        # support = get_support_MTGCN(inputs, L, L3)  #MTGCN
        support = get_support(inputs, L, L3)
        output = torch.mm(support.to(device), self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_layers, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        if n_layers == 1:
            self.first_layer = GraphConvolution(n_features, n_classes, dropout)
        else:
            self.first_layer = GraphConvolution(n_features, hidden_dim, dropout)
            # self.last_layer = GraphConvolution(hidden_dim, n_classes, dropout) # MTGCN

            self.last_layer = GraphConvolution(hidden_dim*3, n_classes, dropout)
            # self.last_layer = GraphConvolution(hidden_dim*2, n_classes, dropout)  # 只用三阶

            if n_layers > 2:
                self.gc_layers = nn.ModuleList([
                    # GraphConvolution(hidden_dim, hidden_dim, 0) for _ in range(n_layers - 2) # MTGCN

                    GraphConvolution(hidden_dim*3, hidden_dim, 0) for _ in range(n_layers - 2)
                    # GraphConvolution(hidden_dim*2, hidden_dim, 0) for _ in range(n_layers - 2)  # 只用三阶
                    # GraphConvolution(hidden_dim*3, hidden_dim*3, 0) for _ in range(n_layers - 2)

                ])
                # self.last_layer = GraphConvolution(hidden_dim, n_classes, dropout)   # MTGCN

                # self.last_layer = GraphConvolution(hidden_dim*2, n_classes, dropout)  # 只用三阶
                self.last_layer = GraphConvolution(hidden_dim*3, n_classes, dropout)  
                # self.last_layer = GraphConvolution(hidden_dim*9, n_classes, dropout)  
            
        self.n_layers = n_layers
        self.relu = nn.ReLU()
    

    def forward(self, inputs, L, L3):
        if self.n_layers == 1:
            x = self.first_layer(inputs, L, L3)
        else:
            x = self.relu(self.first_layer(inputs, L, L3))

            if self.n_layers > 2:
                for i, layer in enumerate(self.gc_layers):
                    x = self.relu(layer(x, L, L3))
            x = self.last_layer(x, L, L3)
        return x
    





