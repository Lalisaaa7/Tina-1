import torch
from modules.model import Enhanced_GCN_with_Attention
from data_loader_from_raw import load_raw_dataset
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F

def test_model(model, test_data, device):
    print("\n📊 Evaluating on test set...")
    model.eval()
    loader = DataLoader(test_data, batch_size=8, shuffle=False)

    total, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
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
        auc = 0.0

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")

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

    return accuracy, f1, mcc, auc


def main():
    print("🧪 Loading test set from ./Raw_data/DNA-46_Test.txt ...")
    all_data = load_raw_dataset('./Raw_data/DNA-46_Test.txt')
    print(f"Loaded {len(all_data)} test samples.")

    if len(all_data) == 0:
        print("❌ No test data found. Please check the file.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ⚠ 注意：确保参数和训练时一致
    in_dim = 1280
    model = Enhanced_GCN_with_Attention(
        in_channels=in_dim,
        hidden_channels=64,
        gcn_out_channels=64,
        mlp_hidden=128,
        out_classes=2
    ).to(device)

    try:
        model.load_state_dict(torch.load('./Weights/best_model.pt'))
        model.eval()
        print("✅ Loaded model weights successfully.")
    except FileNotFoundError:
        print("❌ Model weights not found at './Weights/best_model.pt'")
        return

    test_model(model, all_data, device)


if __name__ == '__main__':
    main()
