import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        input_size = config.seq_len
        num_experts = 2
        encoder_hidden_size = 128

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False), nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out