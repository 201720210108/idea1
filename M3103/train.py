import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from model.architecture import TelecomFraudDetector
from model.sampling import LayerwiseNegativeSampler
from model.loss import InfoNCELoss
from torch_geometric.loader import NeighborLoader

def load_data(device):
    """
    加载北京数据集
    Returns: PyTorch Geometric Data object
    """
    try:
        data_path = "论文思路/北京数据集"
        
        # 读取节点特征
        node_features = {}  # 使用字典存储节点特征
        with open(f"{data_path}/TF.features", 'r') as f:
            for line in f:
                values = line.strip().split()
                node_id = int(values[0])  # 第一列是节点ID
                features = [float(x) for x in values[1:]]  # 其余列是特征
                node_features[node_id] = features
        
        # 确保特征按节点ID顺序排列
        sorted_node_ids = sorted(node_features.keys())
        features = [node_features[node_id] for node_id in sorted_node_ids]
        x = torch.FloatTensor(features)
        
        # 创建节点ID映射（从原始ID到连续索引）
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        
        # 读取边列表
        edges = []
        with open(f"{data_path}/TF.edgelist", 'r') as f:
            for line in f:
                if line.strip():  # 跳过空行
                    source, target = map(int, line.strip().split())
                    # 使用映射获取正确的索引
                    if source in node_id_to_index and target in node_id_to_index:
                        source_idx = node_id_to_index[source]
                        target_idx = node_id_to_index[target]
                        edges.append([source_idx, target_idx])
        
        edge_index = torch.LongTensor(edges).t()  # 转置为 [2, num_edges] 格式
        
        # 读取标签
        node_labels = {}  # 使用字典存储节点标签
        with open(f"{data_path}/TF.labels", 'r') as f:
            for line in f:
                node_id, label = map(int, line.strip().split())
                if node_id in node_id_to_index:
                    node_labels[node_id] = label
        
        # 确保标签与节点顺序对应
        labels = [node_labels.get(node_id, 0) for node_id in sorted_node_ids]
        y = torch.LongTensor(labels)
        
        # 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        
        print(f"\nDataset statistics:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of node features: {data.num_features}")
        print(f"Number of classes: {len(torch.unique(y))}")
        print(f"Number of anomalies (label=1): {(y == 1).sum().item()}")
        print(f"Anomaly ratio: {(y == 1).sum().item() / len(y):.4f}")
        print(f"\nEdge index statistics:")
        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge index max value: {edge_index.max().item()}")
        print(f"Edge index min value: {edge_index.min().item()}")
        print(f"Number of unique nodes in edge_index: {len(torch.unique(edge_index))}")
        
        return data.to(device)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train(data, model, optimizer, negative_sampler, criterion, device, num_rounds=3, batch_size=128):
    model.train()
    total_loss = 0
    all_anomaly_scores = []
    
    # 获取节点总数
    num_nodes = data.x.size(0)
    
    # 计算批次数
    num_batches = (num_nodes + batch_size - 1) // batch_size
    
    # 随机打乱节点顺序
    perm = torch.randperm(num_nodes)
    
    for batch_idx in range(num_batches):
        # 获取当前批次的节点索引
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_nodes)
        batch_indices = perm[start_idx:end_idx]
        
        # 获取当前批次的节点特征
        batch_x = data.x[batch_indices]
        
        # 获取与当前批次相关的边
        mask = torch.isin(data.edge_index[0], batch_indices) & torch.isin(data.edge_index[1], batch_indices)
        batch_edge_index = data.edge_index[:, mask]
        
        # 创建节点索引映射
        unique_nodes = torch.unique(batch_indices)
        node_idx_map = {idx.item(): i for i, idx in enumerate(unique_nodes)}
        
        # 重新映射边索引
        new_edge_index = batch_edge_index.clone()
        for i in range(2):
            for j in range(new_edge_index.size(1)):
                new_edge_index[i, j] = node_idx_map[new_edge_index[i, j].item()]
        
        # 检查边索引是否在有效范围内
        if new_edge_index.size(1) > 0:  # 只有当有边时才处理
            max_index = new_edge_index.max().item()
            if max_index >= batch_x.size(0):
                print(f"Warning: Skipping batch {batch_idx} due to invalid edge indices")
                continue
        
        optimizer.zero_grad()
        
        try:
            # Forward pass
            z1, z2 = model(batch_x, new_edge_index)
            
            # Sample negatives
            negative_samples = negative_sampler.sample(
                z1, 
                new_edge_index,
                batch_size=z1.size(0)
            )
            
            # Calculate loss
            loss = criterion(z1, z2, negative_samples)
            
            # Calculate anomaly scores
            scores = model.compute_anomaly_scores(z1, z2, negative_samples)
            
            # 将分数映射回原始节点索引
            batch_scores = torch.zeros(num_nodes, device=device)
            batch_scores[batch_indices] = scores
            all_anomaly_scores.append(batch_scores)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Batch size: {batch_x.size(0)}, Edge index max: {new_edge_index.max().item() if new_edge_index.size(1) > 0 else -1}")
            continue
        
        # 清理内存
        del z1, z2, negative_samples, batch_x, new_edge_index
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f'Processed {batch_idx + 1}/{num_batches} batches')
    
    # 合并所有批次的异常分数
    if all_anomaly_scores:
        final_anomaly_scores = torch.stack(all_anomaly_scores).sum(dim=0)
        final_anomaly_scores = final_anomaly_scores / (final_anomaly_scores.max() + 1e-8)
    else:
        final_anomaly_scores = torch.zeros(num_nodes, device=device)
    
    return total_loss / max(num_batches, 1), final_anomaly_scores

def create_test_subset(data, num_nodes=1000, seed=42):
    """
    从原始数据集创建一个小的测试子集
    Args:
        data: 原始PyTorch Geometric Data对象
        num_nodes: 子集中想要的节点数量
        seed: 随机种子，用于复现结果
    Returns:
        subset_data: 包含子集的新Data对象
    """
    torch.manual_seed(seed)
    
    # 随机选择节点
    all_nodes = torch.arange(data.num_nodes)
    subset_indices = torch.randperm(data.num_nodes)[:num_nodes]
    
    # 获取子集的特征和标签
    subset_x = data.x[subset_indices]
    subset_y = data.y[subset_indices]
    
    # 创建节点索引映射
    node_idx_map = {idx.item(): new_idx for new_idx, idx in enumerate(subset_indices)}
    
    # 获取与子集节点相关的边
    mask = torch.isin(data.edge_index[0], subset_indices) & torch.isin(data.edge_index[1], subset_indices)
    subset_edge_index = data.edge_index[:, mask]
    
    # 重新映射边索引
    new_edge_index = subset_edge_index.clone()
    for i in range(2):
        for j in range(new_edge_index.size(1)):
            new_edge_index[i, j] = node_idx_map[new_edge_index[i, j].item()]
    
    # 创建新的Data对象
    subset_data = Data(
        x=subset_x,
        edge_index=new_edge_index,
        y=subset_y
    )
    
    print(f"\nTest subset statistics:")
    print(f"Number of nodes: {subset_data.num_nodes}")
    print(f"Number of edges: {subset_data.num_edges}")
    print(f"Number of node features: {subset_data.num_features}")
    print(f"Number of anomalies (label=1): {(subset_y == 1).sum().item()}")
    print(f"Anomaly ratio: {(subset_y == 1).sum().item() / len(subset_y):.4f}")
    
    return subset_data

def main():
    # Hyperparameters
    FEATURE_DIM = None
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    TEST_MODE = True  # 添加测试模式标志
    TEST_SUBSET_SIZE = 1000  # 测试子集大小
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    full_data = load_data(device)
    
    # 根据模式选择使用完整数据集还是测试子集
    if TEST_MODE:
        print("\nRunning in test mode with subset of data...")
        data = create_test_subset(full_data, num_nodes=TEST_SUBSET_SIZE)
    else:
        print("\nRunning with full dataset...")
        data = full_data
    
    FEATURE_DIM = data.num_features
    
    print("\nTraining configuration:")
    print(f"Mode: {'Test' if TEST_MODE else 'Full'}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Number of layers: {NUM_LAYERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    # Initialize model and training components
    model = TelecomFraudDetector(
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=1e-5
    )
    
    negative_sampler = LayerwiseNegativeSampler(
        num_samples_per_layer=5,
        layer_weights=[0.8, 0.2]
    )
    criterion = InfoNCELoss()
    
    # 如果是测试模式，减少训练轮数
    if TEST_MODE:
        NUM_EPOCHS = 10
        print(f"Test mode: reduced epochs to {NUM_EPOCHS}")
    
    print("\nStarting training...")
    best_f1 = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        loss, anomaly_scores = train(
            data, model, optimizer, negative_sampler, criterion, 
            device, batch_size=BATCH_SIZE
        )
        
        # 计算评估指标
        if data.y is not None:
            threshold = anomaly_scores.mean() + anomaly_scores.std()
            predicted_anomalies = (anomaly_scores > threshold).float()
            accuracy = (predicted_anomalies == data.y).float().mean()
            
            print(f'Loss: {loss:.4f}, '
                  f'Mean Anomaly Score: {anomaly_scores.mean():.4f}, '
                  f'Accuracy: {accuracy:.4f}')
            
            # 每个epoch都计算详细指标
            tp = ((predicted_anomalies == 1) & (data.y == 1)).sum().item()
            fp = ((predicted_anomalies == 1) & (data.y == 0)).sum().item()
            tn = ((predicted_anomalies == 0) & (data.y == 0)).sum().item()
            fn = ((predicted_anomalies == 0) & (data.y == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 更新最佳模型
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                if not TEST_MODE:  # 只在完整数据集训练时保存模型
                    torch.save(model.state_dict(), 'best_model.pth')
            
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print(f'Number of detected anomalies: {predicted_anomalies.sum().item()}')
            print(f'Best F1: {best_f1:.4f} (Epoch {best_epoch + 1})')

if __name__ == '__main__':
    main() 