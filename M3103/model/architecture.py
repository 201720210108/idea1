import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from .feature_attention import FeatureAttention
from .projection_head import ProjectionHead
from .anomaly_score import AnomalyScorer

class TelecomFraudDetector(nn.Module):
    def __init__(
        self,
        feature_dim=8,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.feature_attention = FeatureAttention(feature_dim)
        self.encoder = GraphEncoder(feature_dim, hidden_dim, num_layers, dropout)
        self.projector = ProjectionHead(hidden_dim)
        self.anomaly_scorer = AnomalyScorer()
        
    def forward(self, x, edge_index):
        # Generate two views
        x1 = x  # Original view
        x2 = self.feature_attention(x)  # Attention-weighted view
        
        # Encode both views
        h1 = self.encoder(x1, edge_index)
        h2 = self.encoder(x2, edge_index)
        
        # Project embeddings
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        return z1, z2
    
    def compute_anomaly_scores(self, z1, z2, negative_samples):
        """计算异常评分"""
        return self.anomaly_scorer(z1, z2, negative_samples)

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        return x 