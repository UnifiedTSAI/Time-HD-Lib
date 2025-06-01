import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from math import sqrt

class Cluster_wise_linear(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList()
        for i in range(n_cluster):
            self.linears.append(nn.Linear(in_dim, out_dim))

        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        bsz = x.shape[0]
        output = []
        for layer in self.linears:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  #[bsz, n_vars,  out_dim, n_cluster]
        prob = prob.unsqueeze(-1)  #[n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).reshape(bsz, -1, self.out_dim)   #[bsz, n_vars, out_dim]
        return output
    
class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster = nn.Linear(d_model*2, 1)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, cluster_emb.shape[-1])      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = int(bn/self.n_vars)
        x_emb_batch = x_emb.repeat(self.n_cluster, 1)   
        cluster_emb_batch = torch.repeat_interleave(cluster_emb, bn, dim=0)
        out = torch.cat([x_emb_batch, cluster_emb_batch], dim=-1)
        prob = F.sigmoid(self.cluster(out)).squeeze(-1).reshape(self.n_cluster, bs, self.n_vars).permute(1,2,0)
        # prob: [bs, n_vars, n_cluster]
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = F.softmax(prob_avg, dim=-1)
        return prob_avg


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        # self.fc3 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class _Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device, epsilon=0.05):
        super(_Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        # linear_layer = [nn.Linear(seq_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        # self.linear = MLP(seq_len, d_model)
        self.linear = nn.Linear(seq_len, d_model)
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device) #nn.Parameter(torch.rand(n_cluster, in_dim * out_dim), requires_grad=True)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        
        
    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_temp)   #[bs*n_vars, n_cluster]
        num_var_pc = torch.sum(mask, dim=0)
        adpat_cluster = torch.matmul(x_emb.transpose(0,1), mask)/(num_var_pc + 1e-6)  #[d_model, n_cluster]
        cluster_emb = cluster_emb + adpat_cluster.transpose(0,1)
        prob_avg = torch.mean(prob, dim=0)      #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        return prob_avg, cluster_emb
    
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern
    
class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', epsilon=0.05):
        super(Cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.device = device
        self.linear = nn.Linear(seq_len, d_model)
        
        # Cluster embeddings
        self.cluster_emb = torch.empty(self.n_cluster, self.d_model).to(device)
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        self.i = 0

    def forward(self, x, cluster_emb):     
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))
        cluster_emb_avg = torch.mean(cluster_emb, dim=0)
        #print(cluster_emb.shape, cluster_emb_.shape, x_emb_.shape, mask.shape)
    
        return prob_avg, cluster_emb_avg
     
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern


def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q


def cluster_aggregator(var_emb, mask):
    '''
        var_emb: (bs*patch_num, nvars, d_model)
        mask: (nvars, n_cluster)
        return: (bs*patch_num, n_cluster, d_model)
    '''
    num_var_pc = torch.sum(mask, dim=0)
    var_emb = var_emb.transpose(1,2)
    cluster_emb = torch.matmul(var_emb, mask)/(num_var_pc + 1e-6)
    cluster_emb = cluster_emb.transpose(1,2)
    return cluster_emb

class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
    
        
        # scores = scores if mask == None else scores * mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
    
        A = A if mask == None else A * mask
        V = torch.einsum("bhls,bshd->blhd", A, values)
    
        return V.contiguous()


class MaskAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(MaskAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out) # B, L, d_model
    
class CrossAttention(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)
       
        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )

        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
       

        return out # B, L, d_model