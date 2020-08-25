import torch
import torch.nn as nn


class MulInteractAttention(nn.Module):

    def __init__(self, hidden_input_size, inter_size):
        super(MulInteractAttention, self).__init__()
        self.hidden_size = hidden_input_size

        self.fc_1 = nn.Linear(hidden_input_size, inter_size, bias=False)
        self.fc_2 = nn.Linear(hidden_input_size, inter_size, bias=False)
        self.fc_out = nn.Linear(inter_size, 1)

    def forward(self, src_hidden, dst_hidden):
        hidden1 = self.fc_1(src_hidden)
        hidden2 = self.fc_2(dst_hidden)
        out = self.fc_out(hidden1 * hidden2)
        return out


class OneHotAttention(nn.Module):

    def __init__(self, n_sources):
        super(OneHotAttention, self).__init__()
        self.att_weight = nn.Linear(n_sources, 1, bias=False)
        self.att_weight.weight = nn.Parameter(torch.ones(size=(1, n_sources)), requires_grad=True)

    def forward(self, hidden):
        out = self.att_weight(hidden)
        return out
