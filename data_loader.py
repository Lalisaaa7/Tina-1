import os
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import esm

# 加载 ESM 模型
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()


def get_esm_features(sequence, model, batch_converter, device='cpu'):
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        residue_representations = token_representations[0, 1:-1].cpu()  # 去除 [CLS] 和 [EOS]

    return residue_representations  # 仅返回特征向量





"""
def sequence_to_graph(sequence, label, k=3, window_size=5):
    x = get_esm_features(sequence, esm_model, batch_converter)
    y = torch.tensor([int(i) for i in label], dtype=torch.long)
    assert len(y) == x.size(0), f"Mismatch: {len(label)} vs {x.size(0)}"

    edge_index = []
    for i in range(len(y)):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < len(y):
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)
"""

def sequence_to_graph(sequence, label, window_size=5):
    # 获取 ESM 表征
    x = get_esm_features(sequence, esm_model, batch_converter)

    y = torch.tensor([int(i) for i in label], dtype=torch.long)
    assert len(y) == x.size(0), f"Mismatch: {len(label)} vs {x.size(0)}"

    L = x.size(0)
    edge_index = []

    # 基于滑动窗口连接邻边（无向图）
    for i in range(L):
        for j in range(i - window_size, i + window_size + 1):
            if 0 <= j < L and i != j:
                edge_index.append([i, j])

    # 构建 edge_index（shape: [2, E]）
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)




def parse_txt_file(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    data_list, buffer = [], []
    for line in lines:
        if line.startswith('>'):
            if len(buffer) == 3:
                name, seq, label = buffer[0][1:], buffer[1], buffer[2]
                if len(seq) == len(label):
                    data = sequence_to_graph(seq, label)
                    data.name = name
                    data.source_file = os.path.basename(path)
                    data_list.append(data)
            buffer = [line]
        else:
            buffer.append(line)
    if len(buffer) == 3:
        name, seq, label = buffer[0][1:], buffer[1], buffer[2]
        if len(seq) == len(label):
            data = sequence_to_graph(seq, label)
            data.name = name
            data.source_file = os.path.basename(path)
            data_list.append(data)
    return data_list




def load_raw_dataset(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = parse_txt_file(file_path)
            all_data.extend(file_data)
    return all_data


def split_dataset_by_filename(all_data):
    train_data = [d for d in all_data if 'Train' in d.source_file]
    test_data = [d for d in all_data if 'Test' in d.source_file]
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    return train_data, val_data, test_data


def load_and_prepare_data(folder, device):
    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)
    train_data = [d.to(device) for d in train_data]
    val_data = [d.to(device) for d in val_data]
    test_data = [d.to(device) for d in test_data]
    feature_dim = train_data[0].x.size(1)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    return train_data, val_data, test_data, feature_dim