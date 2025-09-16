import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import gc
gc.collect()

# 配置类
class ModelConfig:
    def __init__(self):
        # 扩散模型参数
        self.node_dim = 128  # 根据实际ESM特征维度调整
        self.hidden_dim = 256
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.diffusion_conv_type = 'gat'

        # 边生成器参数
        self.edge_hidden_dim = 256
        self.edge_num_layers = 3
        self.edge_conv_type = 'gcn'

        # 训练参数
        self.epochs = 100
        self.batch_size = 2  # 减少批次大小
        self.lr = 1e-3

        # 扩散过程参数
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_timesteps = 500

        # 生成参数
        self.num_new_nodes = 20  # 减少生成节点数量
        self.edge_threshold = 0.3
        self.max_edges_per_node = 4

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
torch.cuda.empty_cache()  # 如果使用 GPU 的话
gc.collect()  # 清理 CPU 内存
device = torch.device('cpu')  # 强制使用CPU

# 配置类
class ModelConfig:
    def __init__(self):
        # 扩散模型参数
        self.node_dim = 128  # 根据实际ESM特征维度调整
        self.hidden_dim = 256
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.diffusion_conv_type = 'gat'

        # 边生成器参数
        self.edge_hidden_dim = 256
        self.edge_num_layers = 3
        self.edge_conv_type = 'gcn'

        # 训练参数
        self.epochs = 100
        self.batch_size = 2  # 减少批次大小
        self.lr = 1e-3

        # 扩散过程参数
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_timesteps = 500

        # 生成参数
        self.num_new_nodes = 20  # 减少生成节点数量
        self.edge_threshold = 0.3
        self.max_edges_per_node = 4


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

                print(f"使用 {max_edges_to_use} 条边训练边生成器")
            else:
                print("没有边，跳过边生成器训练")
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
# 生成新节点并合并到原图
def generate_and_merge(data_list, diffusion_model, edge_generator, diffusion, config, device):
    diffusion_model.eval()
    edge_generator.eval()

    all_new_nodes = []

    # 分批生成新节点
    batch_size = min(100, config.num_new_nodes)
    num_batches = (config.num_new_nodes + batch_size - 1) // batch_size

    for i in range(num_batches):
        current_batch_size = min(batch_size, config.num_new_nodes - i * batch_size)

        with torch.no_grad():
            # 生成新节点
            x = torch.randn(current_batch_size, config.node_dim).to(device)

            """反向扩散过程"""
            # 从最大时间步开始反向去噪
            for t in reversed(range(diffusion.num_timesteps)):
                t_batch = torch.full((current_batch_size,), t, device=device)

                # 确保输入是3D格式
                x_input = x.unsqueeze(0) if x.ndim == 2 else x
                pred_noise = diffusion_model(x_input, t_batch)  # 使用训练好的扩散模型预测噪声
                # 处理输出维度
                if pred_noise.ndim == 3:
                    pred_noise = pred_noise.squeeze(0)
                # 获取当前时间步的参数
                alpha = diffusion.alphas[t]# α_t = 1 - β_t
                alpha_bar = diffusion.alpha_bars[t] # ᾱ_t
                beta = diffusion.betas[t] # β_t
                # 给前几步添加随机噪声
                if t > 0:
                    noise = torch.randn_like(x) # z ~ N(0, I)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
                ) + torch.sqrt(beta) * noise

            all_new_nodes.append(x.cpu())

    new_nodes = torch.cat(all_new_nodes, dim=0)
    new_y = torch.ones(new_nodes.size(0), dtype=torch.long)

    # 合并所有图数据
    all_x = []
    all_y = []
    all_edge_index = []
    node_offset = 0

    for data in data_list:
        # 确保 data 是 Data 对象
        if isinstance(data, Data):
            # 确保x是2维的并且移动到CPU
            x_data = data.x.cpu()  # 确保在CPU上
            if x_data.ndim == 3:
                x_data = x_data.squeeze(0)  # 移除批次维度
                print(f"调整张量维度: 从 3D -> 2D, 形状: {x_data.shape}")
            elif x_data.ndim == 1:
                x_data = x_data.unsqueeze(0)  # 添加特征维度
                print(f"调整张量维度: 从 1D -> 2D, 形状: {x_data.shape}")
            all_x.append(x_data)

            # 确保y是1维的并且移动到CPU
            y_data = data.y.cpu()  # 确保在CPU上
            if y_data.ndim > 1:
                y_data = y_data.squeeze()
            all_y.append(y_data)

            # 调整边索引偏移并移动到CPU
            edges = (data.edge_index + node_offset).cpu()  # 确保在CPU上
            all_edge_index.append(edges)

            node_offset += data.num_nodes
        else:
            # 如果是元组，尝试提取 Data 对象
            try:
                if isinstance(data[0], Data):
                    graph_data = data[0]
                    # 确保x是2维的并且移动到CPU
                    x_data = graph_data.x.cpu()  # 确保在CPU上
                    if x_data.ndim == 3:
                        x_data = x_data.squeeze(0)
                        print(f"调整张量维度: 从 3D -> 2D, 形状: {x_data.shape}")
                    elif x_data.ndim == 1:
                        x_data = x_data.unsqueeze(0)
                        print(f"调整张量维度: 从 1D -> 2D, 形状: {x_data.shape}")
                    all_x.append(x_data)

                    # 确保y是1维的并且移动到CPU
                    y_data = graph_data.y.cpu()  # 确保在CPU上
                    if y_data.ndim > 1:
                        y_data = y_data.squeeze()
                    all_y.append(y_data)

                    # 调整边索引偏移并移动到CPU
                    edges = (graph_data.edge_index + node_offset).cpu()  # 确保在CPU上
                    all_edge_index.append(edges)

                    node_offset += graph_data.num_nodes
                else:
                    print(f"无法识别的数据类型: {type(data)}")
            except:
                print(f"无法处理数据: {type(data)}")

    # 添加新节点（确保是2维的）
    all_x.append(new_nodes)  # new_nodes 已经在CPU上
    all_y.append(new_y)  # new_y 已经在CPU上

    # 创建新节点之间的边（小批量处理）
    new_edges = []
    for i in range(new_nodes.size(0)):
        for j in range(i + 1, min(i + 10, new_nodes.size(0))):  # 只连接附近节点
            if torch.rand(1) < 0.3:  # 随机连接概率
                new_edges.append([node_offset + i, node_offset + j])
                new_edges.append([node_offset + j, node_offset + i])

    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        all_edge_index.append(new_edge_tensor)

    # 确保所有张量都在CPU上并且是2维的
    for i, x_tensor in enumerate(all_x):
        if x_tensor.device != torch.device('cpu'):
            all_x[i] = x_tensor.cpu()
        if x_tensor.ndim != 2:
            print(f"调整张量 {i} 维度: 从 {x_tensor.ndim}D -> 2D")
            if x_tensor.ndim == 3:
                all_x[i] = x_tensor.reshape(-1, x_tensor.shape[-1])
            elif x_tensor.ndim == 1:
                all_x[i] = x_tensor.unsqueeze(-1)

    # 合并所有数据
    try:
        updated_x = torch.cat(all_x, dim=0)
        updated_y = torch.cat(all_y, dim=0)
        updated_edge_index = torch.cat(all_edge_index, dim=1)
    except Exception as e:
        print(f"合并数据时出错: {e}")
        print("各张量形状:", [x.shape for x in all_x])
        print("各张量设备:", [x.device for x in all_x])
        return None

        # 确保节点和标签数量匹配
    if updated_x.size(0) != updated_y.size(0):
        print(f"警告: 节点数 ({updated_x.size(0)}) 和标签数 ({updated_y.size(0)}) 不匹配!")
        print("进行截断处理...")
        min_size = min(updated_x.size(0), updated_y.size(0))
        updated_x = updated_x[:min_size]
        updated_y = updated_y[:min_size]
        print(f"截断后: 节点数={min_size}")

        # 创建新图数据
    updated_data = Data(
        x=updated_x,
        edge_index=updated_edge_index,
        y=updated_y
    )

    print(f"生成 {new_nodes.size(0)} 个新节点")
    print(f"生成 {len(new_edges) // 2} 条新边")
    print(f"节点分布: 负类={sum(updated_y == 0)}, 正类={sum(updated_y == 1)}")
    print(f"总节点数: {updated_x.size(0)}, 总标签数: {updated_y.size(0)}")

    return updated_data
# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()  # 创建配置实例


    # 1. 加载数据
    folder = './Raw_data'
    train_data, val_data, test_data, feature_dim = load_and_prepare_data(folder, device)

    # 2. 特征降维 (使用自编码器)
    reducer = FeatureReducer(input_dim=feature_dim, output_dim=128).to(device)

    # 训练特征降维器
    reducer.train()
    optimizer = torch.optim.Adam(reducer.parameters(), lr=1e-3)

    # 收集所有训练数据用于训练降维器
    all_train_x = []
    for data in train_data:
        all_train_x.append(data.x)
    all_train_x = torch.cat(all_train_x, dim=0).to(device)

    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(all_train_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(50):
        epoch_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            encoded, decoded = reducer(x, return_reconstructed=True)
            loss = F.mse_loss(decoded, x)  # 重建损失
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Reducer Epoch {epoch + 1}/50 - Loss: {epoch_loss / len(loader):.4f}")

    # 应用降维
    def reduce_features(data_list):
        reduced_data = []
        reducer.eval()
        with torch.no_grad():
            for data in data_list:
                reduced_x = reducer(data.x.to(device)).cpu()
                new_data = Data(x=reduced_x, edge_index=data.edge_index, y=data.y)
                new_data.name = data.name
                new_data.source_file = data.source_file
                reduced_data.append(new_data)
        return reduced_data

    train_data = reduce_features(train_data)
    val_data = reduce_features(val_data)
    test_data = reduce_features(test_data)
    feature_dim = 128  # 更新特征维度

    # 3. 训练扩散模型和边生成器
    print("训练扩散模型和边生成器...")

    # 创建训练图列表
    train_graphs = []
    for data in train_data:
        # 确保 data 是 Data 对象
        if isinstance(data, Data):
            train_graphs.append(data)
        else:
            # 如果是元组，尝试提取 Data 对象
            try:
                if isinstance(data[0], Data):
                    train_graphs.append(data[0])
                else:
                    print(f"无法识别的数据类型: {type(data)}")
            except:
                print(f"无法处理数据: {type(data)}")

    # 检查是否有有效数据
    if not train_graphs:
        print("没有有效的训练数据")
        return

    diffusion_model, edge_generator, diffusion = train_diffusion_and_edge_models(
        train_graphs, config, device
    )

    # 4. 生成新节点并合并
    print("生成新节点并合并到训练图...")
    augmented_train_graph = generate_and_merge(
        train_graphs, diffusion_model, edge_generator, diffusion, config, device
    )

    if augmented_train_graph is None:
        print("数据合并失败，使用原始训练数据")
        train_data = train_graphs
    else:
        # 验证合并后的数据
        if augmented_train_graph.x.size(0) != augmented_train_graph.y.size(0):
            print("合并后的数据仍然有问题，使用原始训练数据")
            train_data = train_graphs
        else:
            train_data = [augmented_train_graph]
            print(f"使用增强后的数据: {augmented_train_graph.num_nodes} 个节点")

    # 5. 训练GAT模型
    print("训练GAT模型...")
    model = train_model(train_data, feature_dim, device)
    # 5. 训练GAT模型
    print("训练GAT模型...")
    model = train_model(train_data, feature_dim, device)

    # 6. 测试模型
    print("测试模型...")
    test_metrics = test_model(model, test_data, device)
    print(f"测试结果: 准确率={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()