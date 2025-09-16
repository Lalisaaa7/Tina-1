import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusionGenerator(nn.Module):
    def __init__(self, input_dim, noise_dim=32):
        super(SimpleDiffusionGenerator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 64)
        self.fc2 = nn.Linear(64, input_dim)

    def forward(self, noise):
        x = F.relu(self.fc1(noise))
        return torch.sigmoid(self.fc2(x))

class DiffusionModel:
    def __init__(self, input_dim, device='cpu'):
        self.generator = SimpleDiffusionGenerator(input_dim).to(device)
        self.device = device

    def train_on_positive_samples(self, all_data, batch_size=256, epochs=10):
        """
        用已有正类样本训练生成器模型（支持分批训练，防止OOM）
        """
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # 收集所有正类样本特征
        positive_vectors = []
        for data in all_data:
            pos_feats = data.x[data.y == 1]
            if pos_feats.size(0) > 0:
                positive_vectors.append(pos_feats.cpu())  # 先留在 CPU
        if not positive_vectors:
            print("⚠ 没有可用于训练的正类样本")
            return

        full_pos_data = torch.cat(positive_vectors, dim=0)  # 留在 CPU
        data_size = full_pos_data.size(0)

        print(f" 正类样本总量：{data_size}，使用 batch_size={batch_size} 训练扩散生成器")

        self.generator.train()
        for epoch in range(epochs):
            perm = torch.randperm(data_size)
            epoch_loss = 0
            for i in range(0, data_size, batch_size):
                indices = perm[i:i + batch_size]
                batch_data = full_pos_data[indices].to(self.device)
                noise = torch.randn((batch_data.size(0), 32)).to(self.device)

                generated = self.generator(noise)
                loss = criterion(generated, batch_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Generator Loss: {epoch_loss:.4f}")

            # 清理显存防止碎片堆积
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def generate_positive_sample(self, num_samples=10, batch_size=1024):
        self.generator.eval()
        generated_all = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                noise = torch.randn((current_batch_size, 32)).to(self.device)
                generated = self.generator(noise)
                generated_all.append(generated.cpu())  # 放在 CPU，避免显存爆炸

        return torch.cat(generated_all, dim=0)

