from centralized import load_data, load_model, train_model, test_model
import flwr as fl
import torch
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import matplotlib.pyplot as plt


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, client_id, class_names):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.class_names = class_names
        self.epoch_metrics = {"train_loss": [], "train_accuracy": []}
        self.current_round = 0

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) # OrderedDict: 保證字典的順序和插入順序一致
        self.model.load_state_dict(state_dict, strict=True) # strict=True: 確保模型參數名稱和數量一致

    def fit(self, parameters, config):
        self.current_round += 1 # 紀錄當前回合
        self.set_parameters(parameters)

        # 訓練模型
        train_losses, train_accuracies = train_model(self.model, self.train_loader, epochs=5)

        # 評估模型在訓練集上的表現
        val_loss, val_accuracy = test_model(self.model, self.test_loader, self.class_names)
        print(f"Client {self.client_id} - Round {self.current_round} Validation Accuracy: {val_accuracy:.4f}")

        # 保存每個 epoch 的損失和準確率
        self.epoch_metrics["train_loss"].extend(train_losses)
        self.epoch_metrics["train_accuracy"].extend(train_accuracies)

        # 繪製圖表
        # self.plot_metrics(train_losses, train_accuracies, self.current_round)

        # Return parameters, size, and metrics
        return self.get_parameters(), len(self.train_loader.dataset), {
            # 回傳 metrics 給伺服器
            "train_loss": train_losses[-1],
            "train_accuracy": train_accuracies[-1],
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # 評估 server 傳來的模型參數在測試集上的表現
        val_loss, val_accuracy = test_model(self.model, self.test_loader, self.class_names)
        print(f"Client {self.client_id} - Server Model Validation Accuracy: {val_accuracy:.4f}")

        return val_loss, len(self.test_loader.dataset), { "Server Model val_loss": val_loss,  "Server Model val_accuracy": val_accuracy}

    def plot_metrics(self, train_losses, train_accuracies, current_round):
        plot_dir = f"plots/client_{self.client_id}"
        os.makedirs(plot_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 水平排列

        ax1.plot(train_losses, label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Over Epochs")
        ax1.legend()

        ax2.plot(train_accuracies, label="Train Accuracy", color="orange")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy Over Epochs")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/client_{self.client_id}_round_{current_round}_metrics.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True) # 定義 data_folder 參數
    parser.add_argument("--client_id", type=int, required=True)  # 定義 client_id 參數
    args = parser.parse_args()

    # 讀取資料
    data, labels, class_names = load_data(args.data_folder)

    # 確保標籤在有效範圍內
    num_classes = len(np.unique(labels))
    # assert np.all(labels >= 0) and np.all(labels < num_classes), "標籤超出範圍"

    # 拆分數據集為 7:3 的訓練集和測試集
    if len(data) > 0:
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    else:
        raise ValueError("數據集為空，請檢查數據文件夾。")

    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")

    # 定義輸出形狀和label數
    input_shape = (2000, X_train.shape[2])  # data 結構 [[0, 1, 2,...], [0, 1, 2,...], [0, 1, 2,...], ..., [0, 1, 2,...]]  0, 1, 2 只是feature，實際上填入的是feature的值， 1 是 channel 數

    # 調整資料格式適應 CNN 輸入 (樣本數, 高度, 寬度, 頻道數)
    X_train = X_train.reshape(X_train.shape[0], 1, 2000, X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, 2000, X_test.shape[2])

    # 創建 PyTorch 模型
    model = load_model(input_shape, num_classes) # to(device) 已經包含在 centralize.py 的 load_model 中

    # 將模型移動到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 創建 DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.long).to(device)), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.long).to(device)), batch_size=128, shuffle=False)

    # 創建並啟動聯邦學習客戶端
    client = FederatedClient(model, train_loader, test_loader, client_id=args.data_folder.split('_')[-1], class_names=class_names)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())