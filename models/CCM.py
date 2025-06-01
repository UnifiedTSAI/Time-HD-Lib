import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.CCM_layers import *
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_cluster = args.n_cluster
        self.d_ff = args.d_ff

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.channels = args.enc_in

        self.Linear_Seasonal = Cluster_wise_linear(self.n_cluster, self.channels, self.seq_len, self.pred_len, device=args.accelerator.device)
        self.Linear_Trend = Cluster_wise_linear(self.n_cluster, self.channels,self.seq_len, self.pred_len, device=args.accelerator.device)
        self.Cluster_assigner = Cluster_assigner(self.channels, self.n_cluster, self.seq_len, self.d_ff, device=args.accelerator.device)
        self.cluster_emb = self.Cluster_assigner.cluster_emb
            

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        
        batch_x = x

        self.cluster_prob, cluster_emb = self.Cluster_assigner(x, self.cluster_emb)

        if self.training:
            self.cluster_emb = nn.Parameter(cluster_emb, requires_grad=True)
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.Linear_Seasonal(seasonal_init, self.cluster_prob)
        trend_output = self.Linear_Trend(trend_init, self.cluster_prob)

        x = seasonal_output + trend_output

        loss_aux = self.similarity_loss_batch(batch_x)
        
        return x.permute(0,2,1), loss_aux
    
    def get_similarity_matrix_update(self, batch_data, sigma=None, use_median=True):
        """
        Compute the similarity matrix between different channels of a time series in a batch.
        The similarity is computed using the exponential function on the squared Euclidean distance
        between mean temporal differences of channels.
        
        Parameters:
            batch_data (torch.Tensor): Input data of shape (batch_len, seq_len, channel).
            sigma (float): Parameter controlling the spread of the Gaussian similarity function.
            
        Returns:
            torch.Tensor: Similarity matrix of shape (channel, channel).
        """
        # batch_len, seq_len, num_channels = batch_data.shape
        # similarity_matrix = torch.zeros((num_channels, num_channels), device=batch_data.device)

        # # Compute point-by-point differences along the sequence length
        # time_diffs = batch_data[:, 1:, :] - batch_data[:, :-1, :]  # Shape: (batch_len, seq_len-1, channel)
        
        # # Compute mean of these differences over batch and sequence length
        # channel_representations = time_diffs.mean(dim=(0, 1))  # Shape: (channel,)
        
        # # Compute pairwise similarity
        # for i in range(num_channels):
        #     for j in range(num_channels):
        #         diff = torch.norm(channel_representations[i] - channel_representations[j]) ** 2
        #         similarity_matrix[i, j] = torch.exp(-diff / (2 * sigma ** 2))

        # return similarity_matrix.to(batch_data.device)
        # Shape: (batch_len, seq_len-1, channel)

        time_diffs = batch_data[:, 1:, :] - batch_data[:, :-1, :]

        # Shape: (channel,)
        channel_repr = time_diffs.mean(dim=(0, 1))
        channel_repr = (channel_repr - channel_repr.mean()) / (channel_repr.std() + 1e-6)

        # Compute pairwise squared differences using broadcasting
        diff_matrix = channel_repr.unsqueeze(0) - channel_repr.unsqueeze(1)  # Shape: (channel, channel)
        squared_diff_matrix = diff_matrix ** 2

        if sigma is None:
            # Extract non-diagonal elements
            c = squared_diff_matrix.shape[0]
            mask = ~torch.eye(c, dtype=torch.bool, device=batch_data.device)
            off_diag_values = squared_diff_matrix[mask]
            
            if use_median:
                sigma = torch.sqrt(torch.median(off_diag_values))
            else:
                sigma = torch.sqrt(torch.mean(off_diag_values))

            # Avoid sigma = 0
            sigma = sigma.clamp(min=1e-5)

        # Apply Gaussian similarity
        similarity_matrix = torch.exp(-squared_diff_matrix / (2 * sigma ** 2))

        return similarity_matrix.to(batch_data.device)
    
    # def similarity_loss_batch(self, batch_data):
    #     def concrete_bern(prob, temp = 0.07):
    #         random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(batch_data.device)
    #         random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
    #         prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
    #         prob_bern = ((prob + random_noise) / temp).sigmoid()
    #         return prob_bern
    #     simMatrix = self.get_similarity_matrix_update(batch_data)
    #     membership = concrete_bern(self.cluster_prob)  #[n_vars, n_clusters]
    #     temp_1 = torch.mm(membership.t(), simMatrix) 
    #     SAS = torch.mm(temp_1, membership)
    #     _SS = 1 - torch.mm(membership, membership.t())
    #     loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
    #     ent_loss = (-self.cluster_prob * torch.log(self.cluster_prob + 1e-15)).sum(dim=-1).mean()
    #     return loss + ent_loss
    def similarity_loss_batch(self, batch_data):
        def concrete_bern(prob, temp=0.07):
            # Sample from Concrete (relaxed Bernoulli) distribution
            eps = 1e-10
            noise = torch.empty_like(prob).uniform_(eps, 1 - eps)
            noise = torch.log(noise) - torch.log(1.0 - noise)
            logit = torch.log(prob + eps) - torch.log(1.0 - prob + eps)
            return ((logit + noise) / temp).sigmoid()

        sim_matrix = self.get_similarity_matrix_update(batch_data)  # [n_vars, n_vars]
        membership = concrete_bern(self.cluster_prob)  # [n_vars, n_clusters]

        # Project similarity through soft cluster assignments
        sim_proj = membership.T @ sim_matrix @ membership  # [n_clusters, n_clusters]
        # Orthogonality-promoting term
        mem_dot = membership @ membership.T  # [n_vars, n_vars]
        orth_loss = 1 - mem_dot

        # Main loss: contrastive between within-cluster similarity and between-cluster dissimilarity
        # loss = -torch.trace(sim_proj) + torch.trace(orth_loss @ sim_matrix) + membership.shape[0]
        
        n_vars = membership.shape[0]

        # Entropy regularization to avoid degenerate assignments
        entropy = (-self.cluster_prob * torch.log(self.cluster_prob + 1e-15)).sum(dim=-1).mean()

        loss = (
            -torch.trace(sim_proj) / n_vars +
            torch.trace(orth_loss @ sim_matrix) / (n_vars ** 2) +
            entropy
        )

        return loss + entropy