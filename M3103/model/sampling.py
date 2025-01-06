import torch
import torch.nn.functional as F

class LayerwiseNegativeSampler:
    def __init__(self, layer_weights=[0.8, 0.2], num_samples_per_layer=10):
        self.layer_weights = layer_weights
        self.num_samples_per_layer = num_samples_per_layer
        
    def sample(self, embeddings, edge_index, batch_size=None):
        device = embeddings.device
        num_nodes = embeddings.size(0)
        
        # 限制计算规模
        if batch_size is not None:
            num_nodes = min(num_nodes, batch_size)
        
        # 分块计算相似度矩阵
        chunk_size = 1024  # 可以根据显存大小调整
        sim_matrix_chunks = []
        
        for i in range(0, num_nodes, chunk_size):
            end_idx = min(i + chunk_size, num_nodes)
            chunk_embeddings = embeddings[i:end_idx]
            
            chunk_sim = torch.mm(chunk_embeddings, embeddings.t())
            chunk_sim = chunk_sim / (
                chunk_embeddings.norm(dim=1)[:, None] @ embeddings.norm(dim=1)[None, :]
            )
            sim_matrix_chunks.append(chunk_sim)
        
        sim_matrix = torch.cat(sim_matrix_chunks, dim=0)
        
        # 构建稀疏邻接矩阵
        edge_index = edge_index.to(device)
        adj = torch.sparse_coo_tensor(
            edge_index, 
            torch.ones(edge_index.size(1), device=device),
            size=(num_nodes, num_nodes)
        )
        
        # 计算二阶邻接矩阵（稀疏格式）
        adj2 = torch.sparse.mm(adj, adj)
        
        negative_samples = []
        for i in range(num_nodes):
            # 获取邻居（使用稀疏操作）
            first_neighbors = torch.nonzero(adj[i].to_dense()).squeeze()
            second_neighbors = torch.nonzero(adj2[i].to_dense()).squeeze()
            
            # 采样过程（保持不变）
            node_samples = self._sample_neighbors(
                i, first_neighbors, second_neighbors,
                sim_matrix[i], num_nodes, device
            )
            
            negative_samples.append(node_samples)
            
            # 定期清理内存
            if (i + 1) % chunk_size == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.stack(negative_samples)
    
    def _sample_neighbors(self, node_idx, first_neighbors, second_neighbors, 
                         similarities, num_nodes, device):
        # 采样逻辑保持不变，但添加内存优化
        node_samples = []
        
        # 处理一阶邻居
        if len(first_neighbors.shape) > 0:
            valid_first = first_neighbors[similarities[first_neighbors] < 0.1]
            if len(valid_first) > 0:
                n_first = int(self.num_samples_per_layer * self.layer_weights[0])
                first_samples = valid_first[torch.randperm(len(valid_first))[:n_first]]
                node_samples.append(first_samples)
        
        # 处理二阶邻居
        if len(second_neighbors.shape) > 0:
            valid_second = second_neighbors[similarities[second_neighbors] < 0.1]
            if len(valid_second) > 0:
                n_second = int(self.num_samples_per_layer * self.layer_weights[1])
                second_samples = valid_second[torch.randperm(len(valid_second))[:n_second]]
                node_samples.append(second_samples)
        
        # 如果样本不足，随机采样
        if not node_samples:
            available_nodes = torch.arange(num_nodes, device=device)
            available_nodes = available_nodes[available_nodes != node_idx]
            random_samples = available_nodes[torch.randperm(len(available_nodes))[:self.num_samples_per_layer]]
            node_samples.append(random_samples)
        
        return torch.cat(node_samples) 