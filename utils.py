import os
import torch

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix


def readTriplets2Id(path, mode, entity2id, relation2id, with_head=False, allow_emerging=True):

    triplets = []
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        
        ent_cnt = len(entity2id)
        rel_cnt = len(relation2id)
        
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            
            if allow_emerging:
                if h not in entity2id:
                    entity2id[h] = ent_cnt
                    ent_cnt += 1
                if t not in entity2id:
                    entity2id[t] = ent_cnt
                    ent_cnt += 1
                if r not in relation2id:
                    relation2id[r] = rel_cnt
                    rel_cnt += 1
            
            triplets.append((entity2id[h], relation2id[r], entity2id[t]))
    
    if allow_emerging:
        return triplets, entity2id, relation2id
    else:
        return triplets


def triplets2HyperRelation_matrix(triplets, entity2id, relation2id):
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    hr_matrix = np.zeros((num_entity, num_relation*2))
    
    for h, r, t in triplets:
        hr_matrix[h][r] += 1
        hr_matrix[t][r+num_relation] += 1
    
    return np.array(hr_matrix)


def hyperGraph2Matrix(hg):
    pass


def sample_neg_triplets(triplets, num_ent, num_neg):
    neg_triplets = torch.LongTensor(triplets).unsqueeze(dim=1).repeat(1, num_neg, 1)
    rand_result = torch.rand((len(triplets), num_neg))
    perturb_head = rand_result < 0.5
    perturb_tail = rand_result >= 0.5
    rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets), num_neg))
    neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
    neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
    return torch.LongTensor(triplets), torch.LongTensor(neg_triplets).view(-1, 3)


def get_rel_info(triplets):
    dic = {}
    for h, r, t in triplets:
        if r not in dic:
            dic[r] = [[h, t], ]
        else:
            dic[r].append([h, t])
    return dic


def move_batch_data_to_device(ent_fearues, adj, rel_adjs, pos_triplets, neg_triplets, device):
    ent_fearues = ent_fearues.to(device)
    adj = adj.to(device)
    rel_adjs = rel_adjs.to(device)
    pos_triplets = pos_triplets.to(device)
    neg_triplets = neg_triplets.to(device)
    
    return ent_fearues, adj, rel_adjs, pos_triplets, neg_triplets


def move_test_data_to_device(ent_fearues, rel_adjs, device):
    ent_fearues = ent_fearues.to(device)
    num_ents = ent_fearues.shape[0]
    ent_idxs = torch.arange(num_ents).to(device)
    # adj = adj.to(device)
    rel_adjs = rel_adjs.to(device)
    
    return ent_fearues, ent_idxs, rel_adjs


def collate_fn(samples):
    return samples


def get_batch_ent_features_and_adj(batch_triplets, num_ent, num_rel):
        adj_matrix = csc_matrix((np.ones(len(batch_triplets)), (batch_triplets[:, 0], batch_triplets[:, 2])), 
                                shape=(num_ent, num_ent)).tocoo()
        row_idxs = adj_matrix.row
        col_idxs = adj_matrix.col
        adj_matrix = torch.Tensor(coo_matrix((np.ones(len(row_idxs)), (row_idxs, col_idxs)), 
                                shape=(num_ent, num_ent)).todense())
        
        # relational adjacent matrices
        relational_adj_matrices = torch.zeros((num_rel, num_ent, num_ent))
        # entity relational feature
        ent_relational_fearues = torch.zeros((num_ent, num_rel * 2))
        for h, r, t in batch_triplets:
            relational_adj_matrices[r][h][t] += 1
            ent_relational_fearues[h][r] += 1
            ent_relational_fearues[t][r + num_rel] += 1

        return ent_relational_fearues, adj_matrix, relational_adj_matrices