import os
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import esm
# ---------- 全局 K-mer 词表 ----------
GLOBAL_VOCAB = {}

esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm_model.eval()
esm_model = esm_model.cuda() if torch.cuda.is_available() else esm_model.cpu()

# 替代 extract_kmer_features
def extract_esm2_features(sequence, model, alphabet):
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens.to(next(model.parameters()).device), repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33][0]

    # 去掉起始/终止符，仅保留真实氨基酸部分
    return token_representations[1:len(sequence)+1]


# ---------- Graph Construction ----------
def sequence_to_graph(sequence, label, k=3, window_size=5):
    x = extract_esm2_features(sequence, esm_model, alphabet)  # 替代原来kmer提特征
    y = torch.tensor([int(i) for i in label], dtype=torch.long)

    num_nodes = x.size(0)
    assert len(y) == num_nodes, f"label length mismatch ({len(label)} vs {num_nodes})"

    edge_index = []
    for i in range(num_nodes):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < num_nodes:
                edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)


# ---------- TXT 文件读取 ----------
def parse_txt_file(path):
    from data_loader_from_raw import sequence_to_graph  # 如果你在同文件中定义就不用加这行
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    data_list = []
    buffer = []

    for line in lines:
        if line.startswith('>'):
            if len(buffer) == 3:
                name = buffer[0][1:]
                seq = buffer[1]
                label = buffer[2]
                if len(seq) == len(label):
                    data = sequence_to_graph(seq, label)
                    data.name = name
                    data.source_file = os.path.basename(path)  # ✅ 添加这句
                    data_list.append(data)
                else:
                    print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")
            buffer = [line]
        else:
            buffer.append(line)

    if len(buffer) == 3:
        name = buffer[0][1:]
        seq = buffer[1]
        label = buffer[2]
        if len(seq) == len(label):
            data = sequence_to_graph(seq, label)
            data.name = name
            data.source_file = os.path.basename(path)  # ✅ 添加这句
            data_list.append(data)
        else:
            print(f"⚠️ Skipping {name}: length mismatch ({len(seq)} vs {len(label)})")

    return data_list


# ---------- 数据加载入口 ----------
def load_raw_dataset(path):
    all_data = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                file_path = os.path.join(path, filename)
                file_data = parse_txt_file(file_path)
                all_data.extend(file_data)
    elif os.path.isfile(path) and path.endswith('.txt'):
        file_data = parse_txt_file(path)
        all_data.extend(file_data)
    else:
        raise ValueError(f"❌ 无效的路径: {path}")
    return all_data


# ---------- 数据集划分逻辑 ----------
def split_dataset_by_filename(all_data):
    print("Available data names:")
    for data in all_data:
        print(data.name)

    # 更改匹配条件，例如只匹配包含 "Train" 或 "Test" 的文件名
    train_data = [data for data in all_data if "Train" in data.name]
    test_data = [data for data in all_data if "Test" in data.name]

    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    if len(train_data) == 0 or len(test_data) == 0:
        print("❌ No matching files found. Please check the data in the 'Raw_data' folder.")
        return [], [], []

    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

    return train_data, val_data, test_data




if __name__ == '__main__':
    folder = './Raw_data'
    data_all = load_raw_dataset(folder)
    train_data, val_data, test_data = split_dataset_by_filename(data_all)
    print(f"Loaded: {len(data_all)} samples")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
