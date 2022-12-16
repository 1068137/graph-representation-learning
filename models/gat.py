import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv


class GraphAttentionNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8, v2=False, dropout=0.6, device="cpu"):
        super(GraphAttentionNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                nn.Dropout(p=dropout).to(device),
                GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads).to(device) if not v2
                else GATv2Conv(in_channels=input_dim, out_channels=hidden_dim, heads=heads).to(device),
                nn.Dropout(p=dropout).to(device),
                nn.ELU().to(device),
                GATConv(in_channels=hidden_dim * heads, out_channels=self.output_dim, heads=1).to(device) if not v2
                else GATv2Conv(in_channels=hidden_dim * heads, out_channels=self.output_dim, heads=1).to(device),
                nn.Dropout(p=dropout).to(device),
                nn.Softmax(dim=-1).to(device)
            ]
        )

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, X, edge_index):
        for layer in self.layers:
            if isinstance(layer, GATConv) or isinstance(layer, GATv2Conv):
                X = layer(X, edge_index)
            else:
                X = layer(X)

        return X
