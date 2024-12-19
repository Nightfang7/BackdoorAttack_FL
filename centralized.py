import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Num GPUs Available: ", torch.cuda.device_count())

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model(model, train_loader, epochs):
    
    # 設定模型為訓練模式
    model.train()

    # 定義 loss function 和 optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # 紀錄每個 epoch 的 loss 和 accuracy
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 累加 loss 和 accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        # 計算每個 epoch 的 loss 和 accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

    return epoch_losses, epoch_accuracies

def test_model(model, test_loader, class_names):
    # 設定模型為評估模式
    model.eval()

    correct, total, total_loss = 0, 0, 0.0 # 初始化變數

    all_preds = []
    all_targets = []

    # 不需要計算梯度
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item() 
            total += target.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 計算平均 loss 和 accuracy
    test_loss = total_loss / len(test_loader)
    accuracy = correct / total
    test_precision = precision_score(all_targets, all_preds, average='weighted')
    val_recall = recall_score(all_targets, all_preds, average='weighted')
    val_f1 = f1_score(all_targets, all_preds, average='weighted')

    # Print classification report for each class
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))

    return test_loss, accuracy

def process_csv_to_vector(df, max_length):
    # 移除 timestamp 欄位
    if 'time_stamp' in df.columns:
        df = df.drop(columns=['time_stamp'])
    
    # 將 DataFrame 轉換為 numpy array
    data = df.to_numpy()

    if data.shape[0] < max_length:
        # 如果資料長度小於 max_length，則補 -2
        padding = np.full((max_length - data.shape[0], data.shape[1]), -2)
        data = np.vstack((data, padding)) # 將 padding 加在資料後面
    elif data.shape[0] > max_length:
        # 如果資料長度大於 max_length，則截取前 max_length 筆資料
        data = data[:max_length, :]

    return data

def preprocess_data(df, max_length=2000, normalize=True):
    # 將資料轉換為 numpy array
    data = process_csv_to_vector(df, max_length) # 同時移除 time_stamp 欄位

    if normalize:
        # 正規化資料 (StandardScaler)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    return data

def load_data(data_folder, max_length=2000, normalize=True):
    all_data = []
    all_labels = []
    class_names = []

    for label, subfolder in enumerate(os.listdir(data_folder)):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            class_names.append(subfolder)
            for file in os.listdir(subfolder_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subfolder_path, file)
                    df = pd.read_csv(file_path)
                    data = preprocess_data(df, max_length, normalize)
                    all_data.append(data)
                    all_labels.append(label)

    X = np.array(all_data)
    y = np.array(all_labels)

    return X, y, class_names
    

def load_model(input_shape, num_classes):
    model = CNN(input_shape, num_classes).to(device)
    return model

import matplotlib.pyplot as plt

def plot_metrics(metrics, filename, xlabel, ylabel, title=None):
    # 確保目錄存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure()
    plt.plot(metrics, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # 設定資料集路徑
    data_folder = "Merge_data_format_remove_ip"
    max_length = 2000

    # 載入資料集
    X, y, class_names = load_data(data_folder, max_length=max_length, normalize=True)

    # 資料集的 shape
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {class_names}")

    # 資料集的類別數量
    num_classes = len(class_names)

    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'訓練集大小: {X_train.shape}')
    print(f'測試集大小: {X_test.shape}')

    # 定義輸出形狀和 label 的數量
    input_shape = (max_length, X_train.shape[2], 1)
    num_classes = len(np.unique(y_train))

    # 將資料轉成 CNN 模型的輸入格式
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

    # 創建 DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=64, shuffle=False)

    # 創建 CNN 模型
    cnn_model = load_model(input_shape, num_classes)

    # 訓練模型
    train_losses, train_accuracies = train_model(cnn_model, train_loader, epochs=10)

    # 評估模型
    test_loss, test_accuracy = test_model(cnn_model, test_loader, class_names)

    # 繪製訓練過程圖表
    plot_metrics(train_losses, "plots/train_loss.png", "Epoch", "Loss", "Training Loss over Epochs")
    plot_metrics(train_accuracies, "plots/train_accuracy.png", "Epoch", "Accuracy", "Training Accuracy over Epochs")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
