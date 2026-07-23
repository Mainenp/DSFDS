import torch
import torch.nn as nn
import torch.nn.functional as F
import src.config as config

class GatedOmicsFusion(nn.Module):
    """Intelligently weights Expression, CNV, and Mutation data using a Learnable Gate."""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(GatedOmicsFusion, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = F.leaky_relu(self.proj(x))
        gates = torch.sigmoid(self.gate(x))
        gated_features = features * gates
        return self.dropout(self.norm(gated_features))

class ResidualBlock(nn.Module):
    """Skip-connection block to preserve gradients in deep networks."""
    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        out = F.leaky_relu(self.norm(self.fc1(x)))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.leaky_relu(out + res)


class TransformerGeneDependencyModel(nn.Module):
    """Global Single-Gene Dependency Predictor using Self-Attention."""
    def __init__(self, num_genes, input_dim=config.OMICS_CHANNELS):
        super(TransformerGeneDependencyModel, self).__init__()
        hidden_dim = config.PARAMS['hidden_dim']
        dropout = config.PARAMS['dropout']

        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)

       
        self.mut_scaler = nn.Parameter(torch.ones(1) * 5.0)

        self.fusion = GatedOmicsFusion(input_dim, hidden_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.PARAMS['transformer_heads'],
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.PARAMS['transformer_layers'])

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def get_embeddings(self, x):
        batch_size, num_genes, _ = x.shape
        device = x.device

        exp_cnv = x[:, :, :2]
        mut = x[:, :, 2:3] * self.mut_scaler
        x_scaled = torch.cat([exp_cnv, mut], dim=-1)

        h_omics = self.fusion(x_scaled)

        gene_indices = torch.arange(num_genes, device=device).unsqueeze(0).expand(batch_size, -1)
        h_identity = self.gene_embedding(gene_indices)

        h = h_omics + h_identity
        h_trans = self.transformer(h)
        return h_trans

    def forward(self, x):
        batch_size, num_genes, _ = x.shape
        device = x.device

        exp_cnv = x[:, :, :2]
        mut = x[:, :, 2:3] * self.mut_scaler
        x_scaled = torch.cat([exp_cnv, mut], dim=-1)

        h_omics = self.fusion(x_scaled)

        gene_indices = torch.arange(num_genes, device=device).unsqueeze(0).expand(batch_size, -1)
        h_identity = self.gene_embedding(gene_indices)

        h = h_omics + h_identity
        h_trans = self.transformer(h)
        out = self.regressor(h_trans).squeeze(-1)
        return out

class SparseGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3, alpha=0.2):
        super(SparseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a_l = nn.Parameter(torch.Tensor(1, 1, num_heads, out_features))
        self.a_r = nn.Parameter(torch.Tensor(1, 1, num_heads, out_features))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.norm = nn.LayerNorm(out_features * num_heads)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)

    def forward(self, x, adj):
        B, N, _ = x.size()
        device = x.device

        # 1. Extract valid edges (Only done on the 153,378 physical PPI edges)
        edges = torch.nonzero(adj > 0, as_tuple=False).t()  # Shape: (2, Num_Edges)
        row, col = edges[0], edges[1]

        # 2. Linear Projection
        h = self.W(x).view(B, N, self.num_heads, self.out_features)  # (B, N, Heads, Features)

        # 3. Extract node features exclusively for the valid edges
        h_row = h[:, row, :, :]  # (B, Num_Edges, Heads, Features)
        h_col = h[:, col, :, :]

        # 4. Calculate Attention Scores for valid edges only
        score_row = (h_row * self.a_l).sum(dim=-1)  # (B, Num_Edges, Heads)
        score_col = (h_col * self.a_r).sum(dim=-1)

        e = self.leakyrelu(score_row + score_col)

        # 5. Sparse Softmax
        e = e - e.max()  # Prevent exploding gradients
        e_exp = torch.exp(e)

        # Sum up the exponential scores for all edges coming into a specific node
        den = torch.zeros(B, N, self.num_heads, device=device, dtype=x.dtype)
        row_exp = row.view(1, -1, 1).expand(B, -1, self.num_heads)
        den.scatter_add_(1, row_exp, e_exp)

        # Divide to get final softmax attention weights
        den_gathered = den.gather(1, row_exp)
        alpha = e_exp / (den_gathered + 1e-16)
        alpha = self.dropout(alpha)

        # 6. Sparse Message Passing
        # Multiply attention weights by the sender node's features
        msg = alpha.unsqueeze(-1) * h_col  # (B, Num_Edges, Heads, Features)

        # Route the weighted messages to the receiver nodes and sum them up
        out = torch.zeros(B, N, self.num_heads, self.out_features, device=device, dtype=x.dtype)
        row_out_exp = row.view(1, -1, 1, 1).expand(B, -1, self.num_heads, self.out_features)
        out.scatter_add_(1, row_out_exp, msg)

        # 7. Concatenate all heads
        out = out.view(B, N, self.num_heads * self.out_features)

        return self.norm(F.gelu(out))


class PPI_MOGAT(nn.Module):
    """
    The Unified Dual-Engine Architecture.
    Inherits features from the pre-trained Transformer to feed into GAT.
    """

    def __init__(self, num_genes, input_dim=3):
        super(PPI_MOGAT, self).__init__()
        hidden_dim = config.PARAMS.get('hidden_dim', 512)  # 自动对齐 Transformer 维度
        dropout = config.PARAMS.get('dropout', 0.3)
        num_heads = 4

        self.transformer_branch = TransformerGeneDependencyModel(num_genes, input_dim)

        for param in self.transformer_branch.parameters():
            param.requires_grad = False

        head_dim = hidden_dim // num_heads
        self.gat1 = SparseGATLayer(hidden_dim, head_dim, num_heads=num_heads, dropout=dropout)
        self.gat2 = SparseGATLayer(hidden_dim, head_dim, num_heads=num_heads, dropout=dropout)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, adj):
        with torch.no_grad():
            h_local = self.transformer_branch.get_embeddings(x)

        h_graph = self.gat1(h_local, adj)
        h_graph = self.gat2(h_graph, adj)

        h_combined = h_graph + h_local

        out = self.predictor(h_combined)
        return out.squeeze(-1)


    def get_node_embeddings(self, x, adj):
        batch_size, num_genes, _ = x.shape
        device = x.device

        exp_cnv = x[:, :, :2]
        mut = x[:, :, 2:3] * self.transformer_branch.mut_scaler
        x_scaled = torch.cat([exp_cnv, mut], dim=-1)

        h_omics = self.transformer_branch.fusion(x_scaled)

        gene_indices = torch.arange(num_genes, device=device).unsqueeze(0).expand(batch_size, -1)
        h_identity = self.transformer_branch.gene_embedding(gene_indices)

        h_in = h_omics + h_identity
        h_local = self.transformer_branch.transformer(h_in)

        h_graph = self.gat1(h_local, adj)
        h_graph = self.gat2(h_graph, adj)

        h_combined = h_graph + h_local

        gene_embeddings = h_combined.mean(dim=0)  # Shape: (Num_Genes, Hidden_Dim)

        return gene_embeddings
