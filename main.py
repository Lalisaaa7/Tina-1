import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from modules.data_loader import load_and_prepare_data
from modules.model import GAT_with_Residual, train_model, test_model
from modules.ddpm_diffusion_model import train_diffusion_model, generate_augmented_data
import gc


# --- 配置类  ---
class ModelConfig:
    def __init__(self):
        self.node_dim = 128
        self.hidden_dim = 256
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.diffusion_conv_type = 'gat'
        self.edge_hidden_dim = 256
        self.edge_num_layers = 3
        self.edge_conv_type = 'gcn'
        self.epochs = 100
        self.batch_size = 2
        self.lr = 1e-3
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_timesteps = 500
        # 这个值将被动态计算
        self.num_new_nodes = 0
        self.edge_threshold = 0.3
        self.max_edges_per_node = 4
        # 新增窗口大小参数
        self.window_size = 5

# FeatureReducer, DiffusionModel, EdgeGenerator, DiffusionProcess, train_diffusion_and_edge_models
# 特征降维器
class FeatureReducer(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=512):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        # 解码器（用于重建损失）
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, return_reconstructed=False):
        encoded = self.encoder(x)
        if return_reconstructed:
            decoded = self.decoder(encoded)
            return encoded, decoded
        return encoded


# 扩散模型去噪网络
class DiffusionModel(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1, conv_type='gat'):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type

        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 输入投影
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GNN层
        if conv_type == 'gat':
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif conv_type == 'gcn':
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif conv_type == 'sage':
            self.gnn_layers = nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])

        # 残差块
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, t, edge_index=None):
        # 处理不同输入维度
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [B, D] -> [1, B, D]
            added_batch_dim = True
        else:
            added_batch_dim = False

        if x.ndim == 2:
            x = x.unsqueeze(0)  # [B, D] -> [1, B, D]
            batch_size, num_nodes, _ = x.shape
        else:
            batch_size, num_nodes, _ = x.shape

        # 时间嵌入
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.unsqueeze(1).expand(-1, num_nodes, -1)

        # 输入投影
        h = self.input_proj(x) + t_embed

        # GNN处理
        for i in range(self.num_layers):
            # 展平处理
            h_flat = h.view(-1, self.hidden_dim)

            # 应用GNN
            if edge_index is not None:
                gnn_out = self.gnn_layers[i](h_flat, edge_index)
                gnn_out = gnn_out.view(batch_size, num_nodes, self.hidden_dim)
                h = h + gnn_out

            # 残差连接
            h = h + self.residual_blocks[i](h)

        # 输出
        result = self.output_proj(h)

        # 恢复原始维度
        if added_batch_dim and result.ndim == 3:
            result = result.squeeze(0)

        return result


# 边生成器
class EdgeGenerator(nn.Module):
    def __init__(self, node_dim, hidden_dim=256, num_layers=3, conv_type='gcn'):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 使用稀疏GNN层
        if conv_type == 'gcn':
            self.gnn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])
        elif conv_type == 'gat':
            self.gnn_layers = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=1) for _ in range(num_layers)  # 减少头数
            ])
        elif conv_type == 'sage':
            self.gnn_layers = nn.ModuleList([
                SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

        # 边预测器 - 改为处理边而不是全连接
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, edge_index=None):
        h = self.node_encoder(node_features)

        # 应用GNN
        if edge_index is not None:
            for layer in self.gnn_layers:
                h = layer(h, edge_index)
                h = F.silu(h)

        # 只计算现有边的概率，而不是全连接
        if edge_index is not None and edge_index.size(1) > 0:
            src, dst = edge_index
            src_features = h[src]
            dst_features = h[dst]
            pair_features = torch.cat([src_features, dst_features], dim=-1)
            edge_probs = self.edge_predictor(pair_features).squeeze()
        else:
            edge_probs = torch.tensor([], device=h.device)

        return edge_probs


# 扩散过程
class DiffusionProcess:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

        # 在指定设备上创建参数
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def diffuse(self, x_0, t):
        """前向扩散过程"""
        # 确保t在正确设备上
        t = t.to(self.device)

        # 索引扩散参数
        alpha_bar = self.alpha_bars[t]
        # 获取预定义的调度参数
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)
        # 生成与原始数据相同形状的随机噪声
        noise = torch.randn_like(x_0)
        # 计算加噪后的数据
        x_t = sqrt_alpha_bar.unsqueeze(-1) * x_0 + sqrt_one_minus_alpha_bar.unsqueeze(-1) * noise
        return x_t, noise

    def sample_timesteps(self, n):
        """随机采样时间步"""
        return torch.randint(0, self.num_timesteps, (n,), device=self.device)


# 训练扩散模型和边生成器
def train_diffusion_and_edge_models(data_list, config, device):
    from torch_geometric.data import Data
    import gc

    # 收集所有训练数据
    all_x = []
    all_y = []
    all_edge_index = []
    node_offset = 0

    for data in data_list:
        if isinstance(data, Data):
            all_x.append(data.x.to(device))
            all_y.append(data.y.to(device))
            all_edge_index.append(data.edge_index + node_offset)
            node_offset += data.num_nodes

    if all_x:
        all_x = torch.cat(all_x, dim=0)
    else:
        all_x = torch.empty((0, config.node_dim), device=device)

    if all_y:
        all_y = torch.cat(all_y, dim=0)
    else:
        all_y = torch.empty((0,), dtype=torch.long, device=device)

    if all_edge_index:
        all_edge_index = torch.cat(all_edge_index, dim=1)
    else:
        all_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    train_graph = Data(x=all_x, edge_index=all_edge_index, y=all_y)

    # 初始化模型
    diffusion_model = DiffusionModel(
        node_dim=config.node_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        conv_type=config.diffusion_conv_type
    ).to(device)

    edge_generator = EdgeGenerator(
        node_dim=config.node_dim,
        hidden_dim=config.edge_hidden_dim,
        num_layers=config.edge_num_layers,
        conv_type=config.edge_conv_type
    ).to(device)

    diffusion = DiffusionProcess(
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        num_timesteps=config.num_timesteps,
        device=device
    )

    diffusion_optim = torch.optim.Adam(diffusion_model.parameters(), lr=config.lr)
    edge_optim = torch.optim.Adam(edge_generator.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        # ------- Diffusion 模型训练 -------
        diffusion_model.train()
        diffusion_optim.zero_grad()

        # 获取正类节点
        pos_nodes = all_x[all_y == 1]
        if pos_nodes.size(0) == 0:
            print("⚠️ 没有正类节点可用于训练扩散模型")
            continue

        batch_size = min(config.batch_size, pos_nodes.size(0))
        idx = torch.randint(0, pos_nodes.size(0), (batch_size,), device=device)
        x_0 = pos_nodes[idx]

        t = diffusion.sample_timesteps(batch_size).to(device)
        x_t, noise = diffusion.diffuse(x_0, t)

        # 确保输入是3D格式 [1, batch_size, node_dim]
        x_input = x_t.unsqueeze(0)
        pred_noise = diffusion_model(x_input, t)

        # 处理输出维度 - 移除批次维度7y
        if pred_noise.ndim == 3:
            pred_noise = pred_noise.squeeze(0)

        # 确保形状完全匹配
        if pred_noise.shape != noise.shape:
            min_batch = min(pred_noise.size(0), noise.size(0))
            pred_noise = pred_noise[:min_batch]
            noise = noise[:min_batch]

        diffusion_loss = F.mse_loss(pred_noise, noise)
        diffusion_loss.backward()
        diffusion_optim.step()

        # ------- 边生成器训练 -------
        edge_generator.train()
        edge_optim.zero_grad()

        skip_edge_train = False
        edge_loss = torch.tensor(0.0, device=device)

        try:
            # 检查是否有足够的边进行训练
            if train_graph.edge_index.size(1) > 0:
                # 随机选择一部分边进行训练，避免内存问题
                max_edges_to_use = min(500, train_graph.edge_index.size(1))
                edge_indices = torch.randperm(train_graph.edge_index.size(1))[:max_edges_to_use]
                selected_edges = train_graph.edge_index[:, edge_indices]

                # 使用选中的边进行训练
                edge_probs = edge_generator(train_graph.x, selected_edges)
                edge_labels = torch.ones_like(edge_probs)
                edge_loss = F.binary_cross_entropy(edge_probs, edge_labels)

                # print(f"使用 {max_edges_to_use} 条边训练边生成器")
            else:
                # print("没有边，跳过边生成器训练")
                skip_edge_train = True

        except RuntimeError as e:
            print(f"⚠️ 边生成器训练跳过（内存不足）: {e}")
            skip_edge_train = True

        if not skip_edge_train:
            edge_loss.backward()
            edge_optim.step()

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(
                f"Epoch {epoch}/{config.epochs} - Diffusion Loss: {diffusion_loss.item():.4f}, Edge Loss: {edge_loss.item():.4f}")

    return diffusion_model, edge_generator, diffusion


def generate_and_merge(data_list, diffusion_model, diffusion, config, device):
    """
    生成新的正类节点，与旧节点合并，并使用滑动窗口为所有节点重建边。
    """
    diffusion_model.eval()

    # 1. 如果不需要生成节点，直接合并原始图并返回
    if config.num_new_nodes <= 0:
        print("无需生成新节点，仅合并原始图。")
        all_x = torch.cat([d.x for d in data_list], dim=0)
        all_y = torch.cat([d.y for d in data_list], dim=0)
        # 需要正确地合并 edge_index
        edge_index_list = []
        node_offset = 0
        for data in data_list:
            edge_index_list.append(data.edge_index + node_offset)
            node_offset += data.num_nodes
        all_edge_index = torch.cat(edge_index_list, dim=1)
        return Data(x=all_x, edge_index=all_edge_index, y=all_y)

    # 2. 生成新的节点特征
    print(f"计划生成 {config.num_new_nodes} 个新节点...")
    all_new_nodes = []
    batch_size = 512  # 可以根据显存调整批大小
    num_batches = (config.num_new_nodes + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="正在生成新节点"):
        current_batch_size = min(batch_size, config.num_new_nodes - i * batch_size)
        with torch.no_grad():
            x = torch.randn(current_batch_size, config.node_dim).to(device)
            for t in reversed(range(diffusion.num_timesteps)):
                t_batch = torch.full((current_batch_size,), t, device=device)
                x_input = x.unsqueeze(0) if x.ndim == 2 else x
                pred_noise = diffusion_model(x_input, t_batch)
                if pred_noise.ndim == 3:
                    pred_noise = pred_noise.squeeze(0)

                alpha = diffusion.alphas[t]
                alpha_bar = diffusion.alpha_bars[t]
                beta = diffusion.betas[t]

                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
                ) + torch.sqrt(beta) * noise
            all_new_nodes.append(x.cpu())

    new_nodes = torch.cat(all_new_nodes, dim=0)
    new_y = torch.ones(new_nodes.size(0), dtype=torch.long)

    # 3. 合并所有旧节点和新节点
    original_x = torch.cat([d.x.cpu() for d in data_list], dim=0)
    original_y = torch.cat([d.y.cpu() for d in data_list], dim=0)

    updated_x = torch.cat([original_x, new_nodes], dim=0)
    updated_y = torch.cat([original_y, new_y], dim=0)

    # 4. 为所有节点（新旧混合）重建边
    print("正在为所有节点重建边...")
    num_total_nodes = updated_x.size(0)
    new_edge_index = []
    window_size = config.window_size

    for i in tqdm(range(num_total_nodes), desc="正在创建边"):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < num_total_nodes:
                new_edge_index.append([i, j])

    updated_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()

    # 5. 创建最终的图数据
    updated_data = Data(x=updated_x, edge_index=updated_edge_index, y=updated_y)

    print("-" * 30)
    print(f"数据增强完成!")
    print(f"原始节点数: {original_x.size(0)}")
    print(f"生成新节点数: {new_nodes.size(0)}")
    print(f"总 节 点 数: {updated_data.num_nodes}")
    print(f"新图总边数: {updated_data.num_edges}")
    neg_count = (updated_y == 0).sum().item()
    pos_count = (updated_y == 1).sum().item()
    print(f"节点分布: 负类={neg_count}, 正类={pos_count} (比例 ≈ 1:{pos_count / neg_count:.2f})")
    print("-" * 30)

    return updated_data


# --- ain 主函数 ---
def main():
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    config = ModelConfig()

    # 1. 加载数据
    folder = './Raw_data'
    train_data, val_data, test_data, feature_dim = load_and_prepare_data(folder, device)

    # 2. 特征降维 (逻辑保持不变)
    print("开始训练特征降维器...")
    reducer = FeatureReducer(input_dim=feature_dim, output_dim=config.node_dim).to(device)
    optimizer = torch.optim.Adam(reducer.parameters(), lr=1e-3)
    all_train_x = torch.cat([data.x for data in train_data], dim=0).to(device)
    dataset = torch.utils.data.TensorDataset(all_train_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(50):  # 降维器训练轮数可以减少
        epoch_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            _, decoded = reducer(x, return_reconstructed=True)
            loss = F.mse_loss(decoded, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Reducer Epoch {epoch + 1}/50 - Loss: {epoch_loss / len(loader):.4f}")

    def reduce_features(data_list):
        reduced_data = []
        reducer.eval()
        with torch.no_grad():
            for data in data_list:
                reduced_x = reducer(data.x.to(device)).cpu()
                new_data = Data(x=reduced_x, edge_index=data.edge_index.cpu(), y=data.y.cpu())
                new_data.name = getattr(data, 'name', '')
                new_data.source_file = getattr(data, 'source_file', '')
                reduced_data.append(new_data)
        return reduced_data

    print("正在对数据集进行特征降维...")
    train_data = reduce_features(train_data)
    val_data = reduce_features(val_data)
    test_data = reduce_features(test_data)
    config.node_dim = train_data[0].x.size(1)  # 更新特征维度
    print(f"特征降维完成，新维度: {config.node_dim}")

    # 3. 计算需要生成的样本数以平衡数据
    all_train_y = torch.cat([d.y for d in train_data], dim=0)
    total_neg = (all_train_y == 0).sum().item()
    total_pos = (all_train_y == 1).sum().item()

    if total_neg > total_pos:
        config.num_new_nodes = total_neg - total_pos
    else:
        config.num_new_nodes = 0

    print(f"当前训练集: 正类={total_pos}, 负类={total_neg}")

    # 4. 训练扩散模型 (逻辑保持不变)
    print("训练扩散模型...")
    diffusion_model, _, diffusion = train_diffusion_and_edge_models(
        train_data, config, device
    )

    # 5. 生成新节点并合并，创建平衡的数据集
    print("生成新节点并创建平衡的训练图...")
    # 注意：这里不再需要 edge_generator
    augmented_train_graph = generate_and_merge(
        train_data, diffusion_model, diffusion, config, device
    )

    if augmented_train_graph is None or augmented_train_graph.num_nodes == 0:
        print("错误：数据增强失败，程序终止。")
        return

    # 6. 训练GAT模型
    print("使用增强后的平衡数据集训练GAT模型...")
    # 将增强图放入一个列表中，以匹配 train_model 的输入格式
    final_train_data = [augmented_train_graph.to(device)]
    # 确保 test_data 也在正确的设备上
    test_data_on_device = [d.to(device) for d in test_data]

    model = train_model(final_train_data, config.node_dim, device)

    # 7. 测试模型
    print("测试模型...")
    test_metrics = test_model(model, test_data_on_device, device)
    print(
        f"最终测试结果: 准确率={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, MCC={test_metrics['mcc']:.4f}, AUC={test_metrics['auc']:.4f}")


if __name__ == '__main__':
    main()
