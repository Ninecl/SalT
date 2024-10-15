import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_dim, att_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_head = num_heads
        self.attention = ScaleDotProductAttention(num_heads)
        self.w_q = nn.Linear(emb_dim, att_dim * num_heads)
        self.w_k = nn.Linear(emb_dim, att_dim * num_heads)
        self.w_v = nn.Linear(emb_dim, att_dim * num_heads)
        self.w_concat = nn.Linear(att_dim * num_heads, emb_dim)


    def forward(self, x, adj):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. do scale dot product to compute similarity
        out = self.attention(q, k, v, adj)
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)
        return out


    def split(self, tensor):
        num_nodes, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(num_nodes, self.n_head, d_tensor).transpose(0, 1)
        return tensor


    def concat(self, tensor):
        # input is 3 dimension tensor [num_heads, num_nodes, d_tensor]
        num_heads, num_nodes, d_tensor = tensor.size()
        d_model = num_heads * d_tensor
        tensor = tensor.transpose(0, 1).contiguous().view(num_nodes, d_model)
        return tensor
    
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self, num_heads):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.alpha = nn.Parameter(torch.ones(num_heads))
        self.beta = nn.Parameter(torch.zeros(num_heads))


    def forward(self, q, k, v, adj, e=1e-12):
        # split heads to save memory
        Z = []
        # input is 3 dimension tensor [num_heads, num_nodes, d_tensor]
        num_heads, num_nodes, d_tensor = k.size()
        for i in range(num_heads):
            q_i, k_i, v_i, adj_i = q[i], k[i], v[i], adj[i]
            # 1. dot product Query with Key^T to compute similarity
            k_i_t = k_i.transpose(0, 1)  # transpose
            score_i = (q_i @ k_i_t) / math.sqrt(d_tensor) # scaled dot product
            # 2. pass them softmax to make [0, 1] range
            # score_i = self.alpha[i] * torch.eye(num_nodes).cuda() + self.beta[i] * self.softmax(score_i + adj_i)
            score_i = self.beta[i] * self.softmax(score_i + adj_i)
            diag_idxs = torch.arange(num_nodes).cuda()
            score_i[diag_idxs, diag_idxs] = torch.ones(num_nodes).cuda() * self.alpha[i]
            # score_i = F.normalize(score_i, p=1, dim=-1) * adj_i
            # 3. multiply with Value
            z = score_i @ v_i
            # z = torch.spmm(score_i, v_i)
            Z.append(z)
        Z = torch.cat(Z).view(num_heads, num_nodes, d_tensor)
        return Z


class LayerNorm(nn.Module):
    
    def __init__(self, emb_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
    

class PositionwiseFeedForward(nn.Module):

    def __init__(self, emb_dim, hidden_dim, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(emb_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class RelationTrans(nn.Module):

    def __init__(self, emb_dim, att_dim, hidden_dim, num_heads, drop_prob):
        super(RelationTrans, self).__init__()
        self.attention = MultiHeadAttention(emb_dim, att_dim, num_heads)
        self.norm1 = LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(emb_dim, hidden_dim, drop_prob=drop_prob)
        self.norm2 = LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(p=drop_prob)


    def forward(self, x, adj):
        # 1. compute self attention
        _x = x
        x = self.attention(x, adj)
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class AdaptiveGate(nn.Module):
    
    def __init__(self, emb_dim, hidden_dim):
        super(AdaptiveGate, self).__init__()
        # Z
        self.W_xz = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.W_hz = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        # R
        self.W_xr = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.W_hr = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        # H
        self.W_xh = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.W_hh = nn.Parameter(torch.Tensor(emb_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        # O
        self.W_o = nn.Parameter(torch.Tensor(hidden_dim, emb_dim))
        self.b_o = nn.Parameter(torch.zeros(emb_dim))
        
        nn.init.xavier_uniform_(self.W_xz, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_hz, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.b_z, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_xr, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_hr, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.b_r, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_xh, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_hh, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.b_h, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_o, gain=nn.init.calculate_gain('relu'))
        
    
    def forward(self, current_embs, previous_embs):
        Z = torch.sigmoid(torch.matmul(current_embs, self.W_xz) + torch.matmul(previous_embs, self.W_hz) + self.b_z)
        R = torch.sigmoid(torch.matmul(current_embs, self.W_xr) + torch.matmul(previous_embs, self.W_hr) + self.b_r)
        H = torch.tanh(torch.matmul(current_embs, self.W_xh) + torch.matmul(R * previous_embs, self.W_hh) + self.b_h)
        M = H + Z * previous_embs
        out_embs = torch.matmul(M, self.W_o) + self.b_o
        return out_embs, M