#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            
            if i == 0:
                out = self.support[i].mm(pre_sup)
            else:
                out += self.support[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out
    
    
    
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, support, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = 200
        self.support = support
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, self.out_features)))
        
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #self.theta=nn.parameter(torch.nn.init.normal_(tensor, mean=0, std=1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        input = input.type(torch.float32)
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        
        #attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.where(self.support > 0, e, zero_vec)
        #attention = torch.where(self.support > self.theta, e, zero_vec)
        attention = F.softmax(attention, dim=1)  #normalization of attention score
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__( self, input_dim, \
                        support,\
                        dropout_rate=0., \
                        num_classes=10, alpha=0., nheads=1):
        super(GCN, self).__init__()
        self.training = True
        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)
        
    
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.dropout(out, 0.5, training=self.training)
        out = self.layer2(out)
        return out
    
    
class GAT(nn.Module):
    def __init__(self, nfeat, support, num_classes, dropout_rate, alpha, nheads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout_rate

        self.attentions = [GraphAttentionLayer(nfeat, support, dropout=dropout_rate, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(200 * nheads, num_classes, dropout=dropout_rate, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        y=x.clone()
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1),y
    
class GCAT(nn.Module):
    def __init__( self, input_dim, \
                        support,\
                        dropout_rate=0., \
                        num_classes=10, alpha=0., nheads=1):
        super(GCAT, self).__init__()
        self.training = True
        self.heads = nheads
        # GraphConvolution
        #self.layer1 = GraphConvolution(200, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = [GraphAttentionLayer(input_dim, support, dropout=dropout_rate, alpha=alpha, concat=True) for _ in range(nheads)]
        self.layer3 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)
        
    
    def forward(self, x):
        out = []
        for i in range(self.heads):
            out.append(self.layer2[i](x))
        out = torch.stack(out)
        #out = F.relu(self.layer1(out))
        out = F.dropout(out, 0.5, training=self.trainin)
        out = self.layer3(out)
        return out
