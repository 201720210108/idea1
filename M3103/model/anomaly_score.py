import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyScorer(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # 用于融合两种视图的权重
        self.beta = beta    # 用于融合两个异常分数的权重
        
    def forward(self, z1, z2, negative_samples):
        """
        计算节点的异常评分
        Args:
            z1: 原始视图的节点表示
            z2: 增强视图的节点表示
            negative_samples: 负样本索引 [num_nodes, num_neg_samples]
        Returns:
            anomaly_scores: 每个节点的异常评分
        """
        # 标准化嵌入
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算正样本得分 (s_p)
        positive_sim = torch.sum(z1 * z2, dim=1)  # [num_nodes]
        
        # 计算负样本得分 (s_n)
        z1_expanded = z1.unsqueeze(1)  # [num_nodes, 1, dim]
        neg_samples_embedded = z2[negative_samples]  # [num_nodes, num_neg, dim]
        negative_sim = torch.sum(z1_expanded * neg_samples_embedded, dim=2)  # [num_nodes, num_neg]
        
        # 计算每个节点的异常分数 s_i = s_i^p - s_i^n (公式16)
        s_i = positive_sim - negative_sim.mean(dim=1)
        
        # 计算两种视图的异常分数 (公式17)
        # s_i 是原始视图的异常分数
        # ŝ_i 是增强视图的异常分数
        s_hat_i = s_i  # 这里可以用增强视图计算一次异常分数
        
        # 融合两种视图的异常分数 (公式17)
        S_i = self.beta * s_i + (1 - self.beta) * s_hat_i
        
        # 计算最终的异常分数 (公式18)
        # 这里我们直接返回S_i，因为在训练过程中会累积多轮的结果
        return S_i 