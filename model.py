import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_add_pool
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
import torch.optim as optim
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class GAT_with_Residual(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.residual = nn.Linear(in_channels, out_channels)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 输出2个类别
        )

    def forward(self, x, edge_index):
        original_x = x
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        residual = self.residual(original_x)
        x = x + residual
        return self.classifier(x)


def train_model(balanced_data, feature_dim, device):
    # 训练逻辑保持不变，使用新的GAT模型
    model = GAT_with_Residual(
        in_channels=feature_dim,
        hidden_channels=64,
        out_channels=64,
        heads=4
    ).to(device)

class Enhanced_GCN_with_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, gcn_out_channels, mlp_hidden=128, out_classes=2, heads=4):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.attn = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True, dropout=0.2)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.gcn2 = GCNConv(hidden_channels * heads, gcn_out_channels)
        self.norm2 = nn.LayerNorm(gcn_out_channels)
        self.dropout = nn.Dropout(0.3)

        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_channels + in_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index):
        original_x = x  # 保留原始 ESM 表征
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.attn(x, edge_index)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)

        x = torch.cat([x, original_x], dim=1)
        out = self.mlp(x)
        return out


class GINE_with_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, gine_out_channels, mlp_hidden=128, out_classes=2):
        super().__init__()
        self.gine1 = GINEConv(in_channels, hidden_channels)
        self.gine2 = GINEConv(hidden_channels, gine_out_channels)

        self.mlp = nn.Sequential(
            nn.Linear(gine_out_channels, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, out_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.gine1(x, edge_index)
        x = F.relu(x)
        x = self.gine2(x, edge_index)
        x = global_add_pool(x, batch)  # 聚合池化操作
        x = self.mlp(x)
        return x


# 训练函数
def train_model(balanced_data, feature_dim, device):
    print("Training model with balanced dataset...")

    # 检查输入数据
    for i, data in enumerate(balanced_data):
        print(f"数据 {i}: 节点数={data.num_nodes}, 标签数={data.y.size(0)}")
        if data.num_nodes != data.y.size(0):
            print(f"警告: 数据 {i} 中节点数和标签数不匹配!")

    loader = DataLoader(balanced_data, batch_size=8, shuffle=True)

    # 使用新的GAT模型
    model = GAT_with_Residual(
        in_channels=feature_dim,
        hidden_channels=64,
        out_channels=64,
        heads=4
    ).to(device)

    # 损失函数和优化器
    weight = torch.tensor([1.0, 5.0]).to(device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )

    best_acc = 0.0
    for epoch in range(100):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)

            # 确保输出和目标的批次大小匹配
            # 确保输出和目标的批次大小匹配
            if out.size(0) != batch.y.size(0):
                print(f"批次 {batch_idx}: 输出和目标的批次大小不匹配! out: {out.size(0)}, target: {batch.y.size(0)}")
                print(f"批次节点数: {batch.num_nodes}")
                # 截断较长的张量
                min_size = min(out.size(0), batch.y.size(0))
                out = out[:min_size]
                batch_y = batch.y[:min_size]
            else:
                batch_y = batch.y
                # 在else分支中定义min_size
                min_size = batch_y.size(0)

            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            # 累加当前批次实际处理的节点数
            total += batch_y.size(0)
            total_loss += loss.item()

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/100 - Loss: {total_loss / len(loader):.4f} - Acc: {acc:.4f}")

        # 更新学习率
        scheduler.step(acc)

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'Weights/best_model.pt')
            print(f"Saved best model at epoch {epoch + 1} (Acc: {acc:.4f})")

    return model
def test_model(model, test_data, device):
    print("\n📊 Evaluating on test set...")
    model.eval()
    loader = DataLoader(test_data, batch_size=16, shuffle=False)

    total, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)  # logits
        probs = F.softmax(out, dim=1)[:, 1]  # 取正类概率
        pred = out.argmax(dim=1)

        correct += (pred == batch.y).sum().item()
        total += batch.num_nodes

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0  # 防止仅含单类时报错

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")

    # ROC 曲线可视化
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        'accuracy': accuracy,
        'f1': f1,
        'mcc': mcc,
        'auc': auc
    }