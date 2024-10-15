import torch

import torch.nn as nn
import torch.nn.functional as F

from model.attention import RelationTrans, AdaptiveGate


class SalT(nn.Module):
    
    def __init__(self, params, entity2id, relation2id):
        super(SaRT, self).__init__()
        
        self.params = params
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_ent = len(entity2id)
        self.num_rel = len(relation2id)
        
        self.ent_dim = params.ent_dim
        self.rel_dim = params.rel_dim
        self.att_dim = params.att_dim
        self.hidden_dim = params.hidden_dim
        
        self.num_layers = params.num_layers
        self.att_drop_prob = params.att_drop_prob
        
        # entity embeddings initialization
        self.memory_cells = torch.zeros((self.num_ent, self.ent_dim), device=params.device, requires_grad=False)
        # relation embeddings
        self.rel_embs = nn.Embedding(self.num_rel, self.rel_dim)
        
        # encoder paramaters
        self.Win = nn.Linear(self.num_rel*2, self.ent_dim, bias=False)
        self.build_RelationTrans_layers()
        self.ag = AdaptiveGate(self.ent_dim, self.ent_dim)
    
    
    def build_RelationTrans_layers(self):
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = RelationTrans(self.ent_dim, self.att_dim, self.hidden_dim, self.num_rel, self.att_drop_prob)
            self.layers.append(layer)
    
    
    def initialize_parameters(self):
        nn.init.xavier_uniform_(self.rel_embs, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.Win, gain=nn.init.calculate_gain('relu'))
        
    
    def forward(self, ent_relational_fearues, batch_ent_idxs, relational_adj_matrices, re_ratio):
        # 1. get embs of all input entities
        ent_input_features = F.normalize(ent_relational_fearues, p=1, dim=-1)
        ent_embs = self.Win(ent_input_features)
        previous_ent_embs = self.memory_cells[batch_ent_idxs]
        # randomly re-initialize
        num_re = int(len(batch_ent_idxs) * re_ratio)
        re_idxs = torch.randperm(len(batch_ent_idxs))[: num_re]
        previous_ent_embs[re_idxs] *= 0
        ent_embs, memory_cells = self.ag(ent_embs, previous_ent_embs)
        # 2. store memory cells
        self.memory_cells[batch_ent_idxs] = memory_cells.clone().detach()
        # 3. calculate RelationTrans
        for i in range(len(self.layers)):
            # adj = (relational_adj_matrices @ torch.matrix_power(adj_matrix, i))
            adj = relational_adj_matrices
            layer = self.layers[i]
            # print(adj)
            ent_embs = layer(ent_embs, adj)
        
        return ent_embs
    
    
    def score(self, triplets, ent_embs):
        # 1. get idxs
        head_idxs = triplets[:, 0]
        rel_idxs = triplets[:, 1]
        tail_idxs = triplets[:, 2]
        # 2. get embs
        h_embs = ent_embs[head_idxs]
        r_embs = self.rel_embs(rel_idxs)
        t_embs = ent_embs[tail_idxs]
        # 3. h * r * t
        scores = torch.sum(h_embs * r_embs * t_embs, dim=-1)
        
        return scores
    
    
    def update(self, num_ent, device):
        num_seen_ents = self.num_ent
        self.num_ent = num_ent
        num_unseen_memory_cells = self.num_ent - num_seen_ents
        unseen_memory_cells = torch.zeros((num_unseen_memory_cells, self.ent_dim), device=device, requires_grad=False)
        self.memory_cells = torch.cat((self.memory_cells, unseen_memory_cells))
        