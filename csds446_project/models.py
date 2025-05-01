import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GATv2Conv, HeteroConv


class AttentionEdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_classes):
        super().__init__()
        self.q_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.edge_linear = torch.nn.Linear(edge_dim, hidden_dim)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, src_x, dst_x, edge_attr, return_attention=False):
        q = self.q_linear(src_x)
        k = self.k_linear(dst_x)
        e = self.edge_linear(edge_attr)

        attn_score = torch.sum(q * k, dim=-1, keepdim=True)
        attn_weight = torch.sigmoid(attn_score)  # [B, 1]

        attn_out = attn_weight * k + (1 - attn_weight) * q
        combined = attn_out + e
        logits = self.output(combined)

        if return_attention:
            return logits, attn_weight.squeeze(-1)  # return weights too
        return logits


class MLBMatchupPredictor(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, edge_dim, num_classes):
        super().__init__()

        self.gat1 = HeteroConv(
            {
                ("pitcher", "faces", "batter"): GATv2Conv(
                    in_channels=(metadata["pitcher"], metadata["batter"]),
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=True,
                ),
                ("batter", "rev_faces", "pitcher"): GATv2Conv(
                    in_channels=(metadata["batter"], metadata["pitcher"]),
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=True,
                ),
            },
            aggr="sum",
        )

        self.gat2 = HeteroConv(
            {
                ("pitcher", "faces", "batter"): GATv2Conv(
                    in_channels=hidden_channels * 2,
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=False,
                ),
                ("batter", "rev_faces", "pitcher"): GATv2Conv(
                    in_channels=hidden_channels * 2,
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=False,
                ),
            },
            aggr="sum",
        )

        self.attn_edge_classifier = AttentionEdgeClassifier(
            hidden_channels, edge_dim, num_classes
        )

    def forward(self, data, return_attention=False):
        x_dict, edge_index_dict, edge_attr_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
        )

        x_dict = self.gat1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.gat2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        edge_index = data["pitcher", "faces", "batter"].edge_index
        src, dst = edge_index[0], edge_index[1]
        src_x = x_dict["pitcher"][src]
        dst_x = x_dict["batter"][dst]
        edge_attr = data["pitcher", "faces", "batter"].edge_attr

        return self.attn_edge_classifier(
            src_x, dst_x, edge_attr, return_attention=return_attention
        )


class TimeEncoding(torch.nn.Module):
    def __init__(self, num_freqs):
        super().__init__()
        self.freqs = torch.nn.Parameter(torch.randn(num_freqs), requires_grad=True)

    def forward(self, t):
        # Expect t shape: [num_edges]
        t = t.unsqueeze(-1)  # [num_edges, 1]
        omega = self.freqs.unsqueeze(0)  # [1, num_freqs]
        return torch.cat(
            [torch.sin(t * omega), torch.cos(t * omega)], dim=-1
        )  # [num_edges, 2*num_freqs]


class AttentionEdgeClassifierTemporal(torch.nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_classes):
        super().__init__()
        self.q_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.edge_linear = torch.nn.Linear(edge_dim, hidden_dim)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, src_x, dst_x, edge_attr, return_attention=False):
        q = self.q_linear(src_x)
        k = self.k_linear(dst_x)
        e = self.edge_linear(edge_attr)

        attn_score = torch.sum(q * k, dim=-1, keepdim=True)
        attn_weight = torch.sigmoid(attn_score)
        attn_out = attn_weight * k + (1 - attn_weight) * q
        combined = attn_out + e
        logits = self.output(combined)

        if return_attention:
            return logits, attn_weight.squeeze(-1)
        return logits


class MLBMatchupPredictorTemporal(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, edge_dim, num_classes, time_dim=16):
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim // 2)
        self.edge_dim_total = edge_dim + time_dim  # edge features + time encodings

        self.gat1 = HeteroConv(
            {
                ("pitcher", "faces", "batter"): GATv2Conv(
                    in_channels=(metadata["pitcher"], metadata["batter"]),
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=self.edge_dim_total,
                    add_self_loops=False,
                    concat=True,
                ),
                ("batter", "rev_faces", "pitcher"): GATv2Conv(
                    in_channels=(metadata["batter"], metadata["pitcher"]),
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=self.edge_dim_total,
                    add_self_loops=False,
                    concat=True,
                ),
            },
            aggr="sum",
        )

        self.gat2 = HeteroConv(
            {
                ("pitcher", "faces", "batter"): GATv2Conv(
                    in_channels=hidden_channels * 2,
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=self.edge_dim_total,
                    add_self_loops=False,
                    concat=False,
                ),
                ("batter", "rev_faces", "pitcher"): GATv2Conv(
                    in_channels=hidden_channels * 2,
                    out_channels=hidden_channels,
                    heads=2,
                    edge_dim=self.edge_dim_total,
                    add_self_loops=False,
                    concat=False,
                ),
            },
            aggr="sum",
        )

        self.attn_edge_classifier = AttentionEdgeClassifierTemporal(
            hidden_channels, self.edge_dim_total, num_classes
        )

    def forward(self, data, return_attention=False):
        x_dict, edge_index_dict, edge_attr_dict = (
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
        )

        # Apply time encoding to edge attributes
        for edge_type in edge_attr_dict:
            timestamps = data[edge_type].edge_time  # [num_edges]
            time_embed = self.time_encoder(timestamps)  # [num_edges, time_dim]
            edge_attr_dict[edge_type] = torch.cat(
                [edge_attr_dict[edge_type], time_embed], dim=-1
            )

        x_dict = self.gat1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.gat2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        edge_index = data["pitcher", "faces", "batter"].edge_index
        src, dst = edge_index[0], edge_index[1]
        src_x = x_dict["pitcher"][src]
        dst_x = x_dict["batter"][dst]
        edge_attr = edge_attr_dict[("pitcher", "faces", "batter")]

        return self.attn_edge_classifier(
            src_x, dst_x, edge_attr, return_attention=return_attention
        )
