import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
from torch_geometric.data import Data
from sklearn.cluster import KMeans
import gc

# --- 1. Diffusion Schedule ---
def get_diffusion_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar

# --- 2. Time Embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.embed = nn.Embedding(T, dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.linear(self.embed(t))

# --- 3. Diffusion Noise Predictor ---
class DiffusionPredictor(nn.Module):
    def __init__(self, input_dim, T):
        super().__init__()
        self.time_embed = TimeEmbedding(T, input_dim)
        self.fc1 = nn.Linear(input_dim * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, input_dim)
        self.shortcut = nn.Linear(input_dim * 2, input_dim)

    def forward(self, x_t, t):
        te = self.time_embed(t).expand_as(x_t)
        x_input = torch.cat([x_t, te], dim=-1)
        h = F.relu(self.fc1(x_input))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        shortcut = self.shortcut(x_input)
        return out + shortcut

# --- 4. Diffusion Model ---
class DiffusionModel:
    def __init__(self, input_dim, T=1000, device='cpu'):
        self.T = T
        self.device = device
        self.predictor = DiffusionPredictor(input_dim, T).to(device)
        self.betas, self.alphas, self.alphas_bar = get_diffusion_schedule(T)
        self.betas = self.betas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)

    def train_on_positive_samples(self, all_data, epochs=20, batch_size=128, lr=1e-3):
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        positive_vectors = []

        # Collect positive samples
        for data in all_data:
            x = data.x.to(self.device)
            y = data.y.to(self.device)
            pos_feats = x[y == 1]
            if pos_feats.size(0) > 0:
                positive_vectors.append(pos_feats)

        if not positive_vectors:
            print("没有可用于训练的正类样本")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0).to(self.device)
        data_size = full_pos_data.size(0)
        print(f"正类样本总量：{data_size}，开始训练扩散模型")

        for epoch in range(epochs):
            perm = torch.randperm(data_size)
            losses = []
            for i in range(0, data_size, batch_size):
                batch = full_pos_data[perm[i:i + batch_size]]
                t = torch.randint(0, self.T, (batch.size(0),), device=self.device)
                eps = torch.randn_like(batch)
                sqrt_ab = torch.sqrt(self.alphas_bar[t])[:, None]
                sqrt_1m_ab = torch.sqrt(1 - self.alphas_bar[t])[:, None]
                x_t = sqrt_ab * batch + sqrt_1m_ab * eps
                eps_pred = self.predictor(x_t, t)
                loss = criterion(eps_pred, eps)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {sum(losses) / len(losses):.4f}")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_positive_sample(self, num_samples=100):
        input_dim = self.predictor.fc3.out_features
        x_t = torch.randn((num_samples, input_dim)).to(self.device)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            beta = self.betas[t]
            alpha = 1 - beta
            ab = self.alphas_bar[t]
            eps_pred = self.predictor(x_t, t_tensor)
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = (1 - alpha) / torch.sqrt(1 - ab)
            mean = coeff1 * (x_t - coeff2 * eps_pred)
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(beta) * noise
            else:
                x_t = mean
        return x_t.cpu()

# --- 5. Entry Point Functions ---
def train_diffusion_model(train_data, feature_dim, device):
    model = DiffusionModel(input_dim=feature_dim, device=device)
    model.train_on_positive_samples(train_data)
    return model

def generate_augmented_data(diff_model, original_data, device, target_ratio=0.5, batch_size=256, top_k=3):
    from torch_geometric.data import Data
    from sklearn.cluster import KMeans
    import torch.nn.functional as F
    import time
    import math

    real_pos_feats = []
    for data in original_data:
        pos_feats = data.x[data.y == 1]
        if pos_feats.size(0) > 0:
            real_pos_feats.append(pos_feats)

    if not real_pos_feats:
        print("❌ 没有真实正类样本，无法进行相似度筛选")
        return []

    real_pos_feats = torch.cat(real_pos_feats, dim=0).to(device)
    real_pos_feats = F.normalize(real_pos_feats, dim=-1)

    total_pos = sum((data.y == 1).sum().item() for data in original_data)
    total_neg = sum((data.y == 0).sum().item() for data in original_data)
    current_ratio = total_pos / (total_pos + total_neg)
    print(f"当前正类比例: {current_ratio:.4f}")

    if current_ratio >= target_ratio:
        print("数据已平衡，无需扩增")
        return []

    total_target_pos = int(target_ratio * (total_neg / (1 - target_ratio)))
    num_to_generate = total_target_pos - total_pos
    print(f"正类样本不足，计划生成 {num_to_generate} 个节点")

    generated_data = []
    start_time = time.time()

    num_batches = math.ceil(num_to_generate / batch_size)
    for i in range(num_batches):
        n = batch_size if (i < num_batches - 1) else num_to_generate - batch_size * (num_batches - 1)

        new_x = diff_model.generate_positive_sample(num_samples=n).to(device)
        new_x = F.normalize(new_x, dim=-1)

        # KMeans 聚类筛选
        kmeans = KMeans(n_clusters=top_k)  # top_k 作为聚类中心数量
        new_x_np = new_x.cpu().numpy()  # 转换为NumPy数组
        kmeans.fit(new_x_np)  # 聚类操作
        cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)  # 转换为PyTorch张量

        # 计算余弦相似度
        cos_sim = torch.matmul(new_x, cluster_centers.T)  # 计算生成样本和簇中心的余弦相似度
        max_sim, _ = cos_sim.max(dim=1)  # 找到每个样本与簇中心的最大相似度
        keep_mask = max_sim > 0.75  # 保留相似度大于0.75的样本

        kept_x = new_x[keep_mask]  # 保留筛选后的样本

        if kept_x.size(0) == 0:
            print(f"批次 {i+1}: 没有通过筛选的样本，跳过该批")
            continue

        edge_index = []
        for j in range(kept_x.size(0) - 1):
            edge_index.append([j, j + 1])
            edge_index.append([j + 1, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        new_y = torch.ones(kept_x.size(0), dtype=torch.long)
        new_data = Data(x=kept_x.cpu(), edge_index=edge_index, y=new_y).to(device)
        new_data.name = f"gen_{i}"
        new_data.source_file = "generated"
        generated_data.append(new_data)

        print(f" 批次 {i + 1}/{num_batches} - 生成并保留 {kept_x.size(0)} 个节点")

    elapsed = time.time() - start_time
    print(f"总计生成 {sum(d.x.size(0) for d in generated_data)} 个正类节点，用时 {elapsed:.2f} 秒")
    return generated_data

