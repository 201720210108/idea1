import torch
import torch.nn as nn

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 使用多层感知机来学习特征注意力权重
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.Softmax(dim=-1)
        )
        
        # 添加特征变换层，增强视图的表示能力
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, x):
        # 计算注意力权重
        weights = self.attention(x)
        
        # 特征变换
        transformed_features = self.feature_transform(x)
        
        # 应用注意力权重到变换后的特征
        return transformed_features * weights 