import flwr as fl
import matplotlib.pyplot as plt
import os
from flwr.server.strategy import FedAvg
import torch
from collections import OrderedDict
import numpy as np
from centralized import load_data, load_model
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def fit_metrics_aggregation_fn(metrics):
    # 聚合訓練指標
    train_losses = [fit_res["train_loss"] for _, fit_res in metrics]
    train_accuracies = [fit_res["train_accuracy"] for _, fit_res in metrics]

    # print("len of train_losses:", len(train_losses))
    # print("len of train_accuracies:", len(train_accuracies))

    aggregated_train_loss = sum(train_losses) / len(train_losses)
    aggregated_train_accuracy = sum(train_accuracies) / len(train_accuracies)

    val_losses = [fit_res["val_loss"] for _, fit_res in metrics]
    val_accuracies = [fit_res["val_accuracy"] for _, fit_res in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # print("val_losses:", val_losses)
    # print("len of val_losses:", len(val_losses))
    # print("len of val_accuracies:", len(val_accuracies))
    # print("Total examples:", total_examples)

    aggregated_val_loss = sum(val_losses) / len(val_losses)
    aggregated_val_accuracy = sum(val_accuracies) / len(val_accuracies)

    return {"train_loss": aggregated_train_loss, "train_accuracy": aggregated_train_accuracy, "val_loss": aggregated_val_loss, "val_accuracy": aggregated_val_accuracy}

def evaluate_metrics_aggregation_fn(metrics):
    # 聚合評估指標
    val_losses = [eval_res["Server Model val_loss"] for num_examples, eval_res in metrics]
    val_accuracies = [eval_res["Server Model val_accuracy"] * num_examples for num_examples, eval_res in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    print("Total examples:", total_examples)
    print("len of val_losses:", len(val_losses))
    print("len of val_accuracies:", len(val_accuracies))

    aggregated_val_loss = sum(val_losses) / len(val_losses)
    aggregated_val_accuracy = sum(val_accuracies) / total_examples
    return {"Server Model val_loss": aggregated_val_loss, "Server Model val_accuracy": aggregated_val_accuracy}


def plot_history(history, output_path="plots/server_history.png"):
    # 提取 rounds
    rounds = range(1, len(history["train_loss"]) + 1)

    # 提取損失和準確率
    train_losses = [loss[1] for loss in history["train_loss"]]
    val_losses = [val_loss[1] for val_loss in history["val_loss"]]
    train_accuracies = [acc[1] for acc in history["train_accuracy"]]
    val_accuracies = [val_acc[1] for val_acc in history["val_accuracy"]]
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 繪製損失和準確率圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 水平排列

    # Plot loss
    ax1.plot(rounds, train_losses, label="Train Loss", marker=".")
    ax1.plot(rounds, val_losses, label="Validation Loss", color="orange", marker=".", linestyle="--")

    ax1.set_xticks(range(2, len(rounds) + 1, 2))  # 設定刻度為整數, 2 步長, len(rounds) + 1 最大顯示刻度, 從 2 開始
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Rounds")
    ax1.legend()

    # Plot accuracy
    ax2.plot(rounds, train_accuracies, label="Train Accuracy", marker=".")
    ax2.plot(rounds, val_accuracies, label="Validation Accuracy", color="orange", marker=".", linestyle="--")

    ax2.set_xticks(range(2, len(rounds) + 1, 2))  # 設定刻度為整數
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy Over Rounds")
    ax2.legend()

    # 存儲圖表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', output_path='plots/confusion_matrix.png'):
    """
    生成並保存混淆矩陣。

    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
        labels: 標籤名稱列表
        title: 圖表標題
        output_path: 輸出文件路徑
    """

    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 確保輸入是 numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 建立完整的混淆矩陣（包含所有類別）
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # 手動計算混淆矩陣
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    # 創建圖表
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, 
                vmin=0, square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 10})
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)

    # 調整版面配置，確保所有元素都能完整顯示
    plt.tight_layout(pad=1.5)
    
    # 保存圖表
    plt.savefig(output_path, bbox_inches='tight', pad_inches=1.0)  # 確保邊緣有足夠空間
    plt.close()

    # 打印分類統計
    total = len(y_true)
    print(f"\n{title}:")
    print("分類結果統計:")
    print(f"總共 {total} 個樣本")
    unique_labels = range(len(labels)) if labels is None else range(len(labels))
    for i in unique_labels:
        count =  np.sum(y_pred == i)
        percentage = count / total * 100
        label = labels[i] if labels else i
        print(f"分類為 {label}: {count} 個樣本 ({percentage:.2f}%)")

    return cm

def plot_performance_comparison(normal_acc, trigger_acc, bsr, output_path='plots/performance_comparison.png'):
    """
    繪製模型性能比較的直方圖。

    Args:
        normal_acc: 正常測試集準確率
        trigger_acc: 觸發器測試集準確率
        bsr: 後門攻擊成功率
        output_path: 輸出文件路徑
    """
    import matplotlib.pyplot as plt

    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設置數據
    metrics = ['Normal Accuracy', 'Trigger Accuracy', 'Backdoor Success Rate']
    values = [normal_acc, trigger_acc, bsr]
    
    # 創建圖表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values)
    
    # 設置樣式
    plt.ylim(0, 1.0)  # 設置y軸範圍從0到1
    plt.ylabel('Rate')
    plt.title('Model Performance Comparison for 30% poisoned data in 1 client')
    
    # 在柱子上添加具體數值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 調整布局
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(output_path)
    plt.close()

def plot_pca_2d(reduced_params, cluster_labels, client_ids, title="PCA Result (2D)", output_path='plots/pca_2d.png'):
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_params[:, 0], reduced_params[:, 1], 
                         c=cluster_labels, cmap='viridis')
    
    # 添加客戶端標籤
    for i, txt in enumerate(client_ids):
        plt.annotate(f'Client {txt}', (reduced_params[i, 0], reduced_params[i, 1]))
    
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

class TestingStrategy(FedAvg):
    def __init__(
        self,
        total_rounds: int = 20,  # 添加總輪數參數
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.total_rounds = total_rounds

    def aggregate_fit(self, server_round, results, failures):
        """在每輪聚合後測試模型表現"""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == self.total_rounds:  # 在最後一輪進行測試
            print("\n開始測試最終模型...")
            
            try:
                # 將參數轉換為 numpy arrays
                parameters = parameters_to_ndarrays(aggregated_parameters)
                
                # 載入測試數據
                normal_test_data, normal_labels, normal_class_names = load_data("testing_normal", normalize=False)
                trigger_test_data, trigger_labels, trigger_class_names = load_data("testing_triggered", normalize=False)
                # print(f"trigger_labels = {trigger_labels}")
                # 獲取 fuerboos 的數據
                fuerboos_idx = trigger_class_names.index('fuerboos')
                fuerboos_mask = (trigger_labels == fuerboos_idx)
                fuerboos_trigger_data = trigger_test_data[fuerboos_mask]
                fuerboos_trigger_labels = trigger_labels[fuerboos_mask]
                print(f"fuerboos_trigger_data = {fuerboos_trigger_data}")
                # print(f"fuerboos_trigger_labels = {fuerboos_trigger_labels}")

                # 獲取 mydoom 的標籤索引，用於後門攻擊成功率計算，mydoom 是目標類別
                mydoom_idx = normal_class_names.index('mydoom')
                # print(f"Class names: {normal_class_names}")
                # print(f"Mydoom index: {mydoom_idx}")

                # 調整數據形狀 (樣本數, channels=1, height=2000, features)
                normal_test_data = normal_test_data.reshape(normal_test_data.shape[0], 1, 2000, normal_test_data.shape[2])
                trigger_test_data = trigger_test_data.reshape(trigger_test_data.shape[0], 1, 2000, trigger_test_data.shape[2])
                fuerboos_trigger_data = fuerboos_trigger_data.reshape(fuerboos_trigger_data.shape[0], 1, 2000, fuerboos_trigger_data.shape[2])
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 準備數據加載器
                normal_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(normal_test_data, dtype=torch.float32).to(device),
                        torch.tensor(normal_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                trigger_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(trigger_test_data, dtype=torch.float32).to(device),
                        torch.tensor(trigger_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                fuerboos_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(fuerboos_trigger_data, dtype=torch.float32).to(device),
                        torch.tensor(fuerboos_trigger_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                # 創建並配置模型
                input_shape = (2000, normal_test_data.shape[3]) # normal_test_data.shape[3] 是 feature 的數量
                num_classes = len(normal_class_names)
                model = load_model(input_shape, num_classes)
                
                # 載入聚合後的參數
                state_dict = OrderedDict()
                for (k, v), param in zip(model.state_dict().items(), parameters):
                    state_dict[k] = torch.tensor(param)
                model.load_state_dict(state_dict)
                
                # 評估模型
                model.eval()
                
                # 評估結果
                normal_acc = test_model(model, normal_loader, class_names=normal_class_names, title='Normal Test Confusion Matrix for 30% poisoned data in 1 client' ,output_path='plots/normal_confusion_matrix.png')
                trigger_acc  = test_model(model, trigger_loader, class_names=normal_class_names, title='Trigger Test Confusion Matrix for 30% poisoned data in 1 client', output_path='plots/trigger_confusion_matrix.png')
                bsr = test_backdoor_success_rate(
                    model, fuerboos_loader, target_class=mydoom_idx, class_names=normal_class_names
                )
                
                # 保存到 metrics
                metrics["final_test_normal"] = normal_acc
                metrics["final_test_trigger"] = trigger_acc
                metrics["final_backdoor_success_rate"] = bsr

                # 繪製性能比較圖
                plot_performance_comparison(normal_acc, trigger_acc, bsr)

                print(f"正常測試集準確率: {metrics['final_test_normal']:.4f}")
                print(f"觸發器測試集準確率: {metrics['final_test_trigger']:.4f}")
                print(f"後門攻擊成功率: {metrics['final_backdoor_success_rate']:.4f}")
                
            except Exception as e:
                print(f"測試過程中出錯: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        return aggregated_parameters, metrics

class DefenseTestingStrategy(FedAvg):
    def __init__(
        self,
        total_rounds: int = 20,  # 添加總輪數參數
        n_components: int = 1,  # PCA components
        n_clusters: int = 2,     # Number of clusters
        **kwargs
    ):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.retained_clients_log = []  # 紀錄每輪保留哪些客戶端
        self.client_id_mapping = {}  # UUID 到數字 ID 的映射
        self.exclusion_history = {}  # UUID 作為 key
        self.total_rounds_completed = 0

    def plot_exclusion_statistics(self, output_path='plots/exclusion_statistics.png'):
        """繪製剔除統計圖"""

        # 確保目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 將 UUID 替換為數字 client_id
        mapped_exclusion_history = {
            self.client_id_mapping[uuid]: count
            for uuid, count in self.exclusion_history.items()
            if uuid in self.client_id_mapping
        }

        plt.figure(figsize=(10, 6))
        # 初始化所有 client_id 和剔除次數
        all_client_ids = sorted(self.client_id_mapping.values())  # 獲取完整的客戶端 ID 列表
        exclusions = [
            mapped_exclusion_history.get(client_id, 0)  # 提取剔除次數，默認為 0
            for client_id in all_client_ids
        ]
        rates = [count / self.total_rounds_completed for count in exclusions]

        bars = plt.bar(all_client_ids, rates)
        
        # 在柱子上添加標籤
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')

        plt.title('Client Exclusion Rates')
        plt.xlabel('Client ID')
        plt.ylabel('Exclusion Rate')
        plt.ylim(0, 1)
        
        # 標記惡意客戶端（假設 client_id=1 是惡意 client）
        # clients = [str(client_id) for client_id in clients]
        malicious_client_id = "1" 
        # if malicious_client_id in clients:
        plt.axvline(x=malicious_client_id, color='r', linestyle='--', alpha=0.3, label='Malicious Client')
        plt.xticks(all_client_ids, [str(client_id) for client_id in all_client_ids])  # 設置 x 軸顯示所有 client ID
        plt.legend()
        
        plt.savefig(output_path)
        plt.close()    
    def _parameters_to_vector(self, parameters):
        """Convert parameters to a single vector"""
        return np.concatenate([p.flatten() for p in parameters])
        
    def _detect_malicious_clients(self, parameters_list, server_round):
        """使用 PCA 和聚類來檢測惡意客戶端"""
        # 轉換各客戶端的參數為向量
        param_vectors = np.array([
            self._parameters_to_vector(params)
            for params in parameters_list
        ])
        print("Client parameters order:")
        for i, _ in enumerate(param_vectors):
            print(f"Position {i}")  # 對應的位置

        print(f"Original parameters shape: {param_vectors.shape}")

        # 計算可用的最大 PCA 維度
        n_samples = param_vectors.shape[0]  # 客戶端數量
        max_components = min(n_samples - 1, self.n_components)  # 確保維度小於樣本數
        
        # Apply PCA
        pca = PCA(n_components=max_components)
        reduced_params = pca.fit_transform(param_vectors)
        
        print(f"After PCA shape: {reduced_params.shape}")
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        
        # 使用 KMeans clustering 來檢測惡意客戶端
        n_clusters = min(self.n_clusters, n_samples)
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(reduced_params)

        # 視覺化 PCA 結果
        client_ids = list(range(len(parameters_list)))
        
        if max_components >= 2:
            plot_pca_2d(reduced_params, cluster_labels, client_ids, 
                   title=f"PCA Result Round {server_round}", output_path=f"plots/pca_2D/pca_result_round{server_round}.png")

        # 找出最大的集群，將其視為良性客戶端
        cluster_sizes = np.bincount(cluster_labels)
        benign_cluster = np.argmax(cluster_sizes)
        
        print(f"Round {server_round} - Cluster sizes: {cluster_sizes}")
        print(f"Round {server_round} - Benign cluster: {benign_cluster}")
        
        # 回傳良性客戶端的索引
        return [i for i, label in enumerate(cluster_labels) if label == benign_cluster]
    
    def _detect_malicious_clients_by_distance(self, parameters_list, server_round):
        param_vectors = np.array([
            self._parameters_to_vector(params)
            for params in parameters_list
        ])
        
        print(f"Original parameters shape: {param_vectors.shape}")
        
        # 計算每對客戶端之間的距離
        distances = np.zeros((len(param_vectors), len(param_vectors)))
        for i in range(len(param_vectors)):
            for j in range(len(param_vectors)):
                distances[i, j] = np.linalg.norm(param_vectors[i] - param_vectors[j])
        
        print(f"Distance matrix:\n{distances}")
        
        # 計算每個客戶端與其他客戶端的平均距離
        avg_distances = np.mean(distances, axis=1)
        print(f"Average distances: {avg_distances}")
        
        # 找出距離最大的客戶端（可能是惡意的）
        potential_malicious = np.argmax(avg_distances)
        
        # 返回其他客戶端的索引
        benign_indices = [i for i in range(len(param_vectors)) if i != potential_malicious]
        
        print(f"Round {server_round} - Potential malicious client: {potential_malicious}")
        
        return benign_indices

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model parameters with defense mechanism and testing"""
        if not results:
            return None, {}
        
        # 記錄原始的 metrics
        original_metrics = {
            "all_clients": [(res.num_examples, res.metrics) for _, res in results]
        }
        # 提取參數和對應的 client ID
        parameters_with_ids = []

        for client_proxy, fit_res in results:
            # 從 fit_res.metrics 中獲取客戶端的實際 ID
            actual_client_id = fit_res.metrics.get("client_id")  # 需要在 client 端傳送這個資訊
            if client_proxy.cid not in self.client_id_mapping:
                # 使用客戶端實際的 ID（0、1、2）而不是自動生成編號
                self.client_id_mapping[client_proxy.cid] = actual_client_id
            
            parameters_with_ids.append((client_proxy.cid, parameters_to_ndarrays(fit_res.parameters)))
            print(f"Client UUID: {client_proxy.cid} -> Actual ID: {actual_client_id}")

        # 分離 ID 和參數
        client_ids = [cid for cid, _ in parameters_with_ids]
        parameters_list = [params for _, params in parameters_with_ids]
        
        # Detect benign clients
        benign_indices = self._detect_malicious_clients(parameters_list, server_round)
        
        # 找出對應的 client IDs
        benign_client_ids = [client_ids[i] for i in benign_indices]
        print(f"Kept client IDs: {benign_client_ids}")

        # 找出被排除的客戶端 ID
        excluded_client_ids = set(client_ids) - set(benign_client_ids)

        # 根據 client ID 過濾原始結果
        filtered_results = [
            result for result in results 
            if result[0].cid in benign_client_ids
        ]

        # 更新記錄剔除的客戶端次數（使用實際的 client ID）
        for cid in excluded_client_ids:
            if cid not in self.exclusion_history:
                self.exclusion_history[cid] = 0
            self.exclusion_history[cid] += 1
        
        self.total_rounds_completed = server_round

        # Log 保留的客戶端
        self.retained_clients_log.append({
            "round": server_round,
            "retained_clients": benign_client_ids,  # 使用實際的 client ID
        })

        # 如果是最後一輪，打印統計信息
        if server_round == self.total_rounds:
            print("\n防禦檢測統計:")
            for client_id, exclusion_count in self.exclusion_history.items():
                exclusion_rate = exclusion_count / self.total_rounds_completed
                print(f"Client {client_id}:")
                print(f"  被剔除次數: {exclusion_count}")
                print(f"  被剔除比率: {exclusion_rate:.2%}")
                if str(client_id) == "1":  # 惡意客戶端的 ID
                    print(f"  檢測成功率: {exclusion_rate:.2%}")

        # 在最後一輪時調用
        if server_round == self.total_rounds:
            self.plot_exclusion_statistics()
            
        if not filtered_results:
            print("Warning: All clients were classified as malicious!")
            return None, {}

        print(f"Round {server_round}: Kept {len(benign_indices)} out of {len(results)} clients")
        print(f"Kept client indices: {benign_client_ids}")

        # 記錄被剔除的 client 的 metrics
        excluded_metrics = {
            "excluded_clients": [
                (results[i][1].num_examples, results[i][1].metrics) 
                for i, result in enumerate(results) 
                if result[0].cid in excluded_client_ids
            ]
        }
        
        # Aggregate parameters from benign clients
        aggregated_parameters, metrics = super().aggregate_fit(server_round, filtered_results, failures)

        # 添加額外的統計信息
        metrics.update({
            "excluded_count": len(excluded_client_ids),
            "total_clients": len(client_ids),
            "kept_clients": len(benign_client_ids),
            "excluded_client_ids": list(excluded_client_ids)
        })
        
        # 在最後一輪進行測試
        if aggregated_parameters is not None and server_round == self.total_rounds:
            print("\n開始測試最終模型...")
            try:
                # 將參數轉換為 numpy arrays
                parameters = parameters_to_ndarrays(aggregated_parameters)
                
                # 載入測試數據
                normal_test_data, normal_labels, normal_class_names = load_data("testing_normal", normalize=False)
                trigger_test_data, trigger_labels, trigger_class_names = load_data("testing_triggered", normalize=False)
                # print(f"trigger_labels = {trigger_labels}")
                # 獲取 fuerboos 的數據
                fuerboos_idx = trigger_class_names.index('fuerboos')
                fuerboos_mask = (trigger_labels == fuerboos_idx)
                fuerboos_trigger_data = trigger_test_data[fuerboos_mask]
                fuerboos_trigger_labels = trigger_labels[fuerboos_mask]
                print(f"fuerboos_trigger_data = {fuerboos_trigger_data}")
                # print(f"fuerboos_trigger_labels = {fuerboos_trigger_labels}")

                # 獲取 mydoom 的標籤索引，用於後門攻擊成功率計算，mydoom 是目標類別
                mydoom_idx = normal_class_names.index('mydoom')
                # print(f"Class names: {normal_class_names}")
                # print(f"Mydoom index: {mydoom_idx}")

                # 調整數據形狀 (樣本數, channels=1, height=2000, features)
                normal_test_data = normal_test_data.reshape(normal_test_data.shape[0], 1, 2000, normal_test_data.shape[2])
                trigger_test_data = trigger_test_data.reshape(trigger_test_data.shape[0], 1, 2000, trigger_test_data.shape[2])
                fuerboos_trigger_data = fuerboos_trigger_data.reshape(fuerboos_trigger_data.shape[0], 1, 2000, fuerboos_trigger_data.shape[2])
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 準備數據加載器
                normal_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(normal_test_data, dtype=torch.float32).to(device),
                        torch.tensor(normal_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                trigger_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(trigger_test_data, dtype=torch.float32).to(device),
                        torch.tensor(trigger_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                fuerboos_loader = DataLoader(
                    TensorDataset(
                        torch.tensor(fuerboos_trigger_data, dtype=torch.float32).to(device),
                        torch.tensor(fuerboos_trigger_labels, dtype=torch.long).to(device)
                    ),
                    batch_size=32,
                    shuffle=False
                )
                
                # 創建並配置模型
                input_shape = (2000, normal_test_data.shape[3]) # normal_test_data.shape[3] 是 feature 的數量
                num_classes = len(normal_class_names)
                model = load_model(input_shape, num_classes)
                
                # 載入聚合後的參數
                state_dict = OrderedDict()
                for (k, v), param in zip(model.state_dict().items(), parameters):
                    state_dict[k] = torch.tensor(param)
                model.load_state_dict(state_dict)
                
                # 評估模型
                model.eval()
                
                # 評估結果
                normal_acc = test_model(model, normal_loader, class_names=normal_class_names, title='Normal Test Confusion Matrix for 30% poisoned data in 1 client' ,output_path='plots/normal_confusion_matrix.png')
                trigger_acc  = test_model(model, trigger_loader, class_names=normal_class_names, title='Trigger Test Confusion Matrix for 30% poisoned data in 1 client', output_path='plots/trigger_confusion_matrix.png')
                bsr = test_backdoor_success_rate(
                    model, fuerboos_loader, target_class=mydoom_idx, class_names=normal_class_names
                )
                
                # 保存到 metrics
                metrics["final_test_normal"] = normal_acc
                metrics["final_test_trigger"] = trigger_acc
                metrics["final_backdoor_success_rate"] = bsr

                # 繪製性能比較圖
                plot_performance_comparison(normal_acc, trigger_acc, bsr)

                print(f"正常測試集準確率: {metrics['final_test_normal']:.4f}")
                print(f"觸發器測試集準確率: {metrics['final_test_trigger']:.4f}")
                print(f"後門攻擊成功率: {metrics['final_backdoor_success_rate']:.4f}")
                
            except Exception as e:
                print(f"測試過程中出錯: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        return aggregated_parameters, metrics

    def save_retained_clients_log(self, filepath="retained_clients_log.json"):
        """Save the log of retained clients to a JSON file."""
        import json
        with open(filepath, "w") as f:
            json.dump(self.retained_clients_log, f, indent=4)

def test_model(model, test_loader, class_names='Normal Test Confusion Matrix', title = None, output_path=None):
    """評估模型準確率"""
    model.eval() # 以防萬一，確保模型處於評估模式
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 計算準確率
    accuracy = sum(1 for x, y in zip(all_preds, all_targets) if x == y) / len(all_preds)
    
    # 生成混淆矩陣
    if class_names:
        cm = plot_confusion_matrix(
            all_targets, 
            all_preds,
            labels=class_names,
            title=title,
            output_path=output_path
        )
    
    return accuracy

def test_backdoor_success_rate(model, trigger_loader, target_class, class_names=None):
    """評估後門攻擊成功率"""
    model.eval()
    all_preds = []
    total = 0
    
    with torch.no_grad():
        for data, _ in trigger_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            total += data.size(0)
    
    # 創建期望的標籤（全部應該是目標類別）
    expected_labels = [target_class] * total
    
    # 生成混淆矩陣
    if class_names:
        cm = plot_confusion_matrix(
            expected_labels,
            all_preds,
            labels=class_names,
            title='Backdoor Attack Confusion Matrix for 30% poisoned data in 1 client',
            output_path='plots/backdoor_confusion_matrix.png'
        )
    
    # 計算成功率
    bsr = sum(1 for x in all_preds if x == target_class) / total
    # print(f"Backdoor Success Rate (BSR): {bsr:.4f}")
    
    return bsr

if __name__ == "__main__":

    num_rounds=20

    # strategy = FedAvg(
    #     fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    #     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    # )

    # strategy = TestingStrategy(
    #     total_rounds=num_rounds,
    #     fraction_fit=1.0,
    #     min_fit_clients=3,
    #     min_available_clients=3,
    #     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    #     fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    # )

    strategy = DefenseTestingStrategy(
        total_rounds=num_rounds,
        n_components=2,  # PCA 維度
        n_clusters=2,     # 良性和惡意兩類
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)

    # 開始 Flower 伺服器
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=server_config,
        strategy=strategy,
    )
    
    # print("type(history): ", type(history))
    # print("dir(history): ", dir(history))
    # print("history: ", history)

    # print(history.losses_distributed)
    # print(history.metrics_distributed)

    # print("Losses Distributed:", history.losses_distributed)
    # print("Metrics Distributed:", history.metrics_distributed)

    # 提取伺服器的損失和準確率
    extracted_history = {
        "train_loss": history.metrics_distributed_fit["train_loss"],  # Train Loss
        "train_accuracy": history.metrics_distributed_fit["train_accuracy"],  # Train Accuracy
        "val_loss": history.metrics_distributed["Server Model val_loss"],  # Validation Loss
        "val_accuracy": history.metrics_distributed["Server Model val_accuracy"],  # Validation Accuracy
        # "val_loss": history.metrics_distributed_fit["val_loss"],  # Validation Loss
        # "val_accuracy": history.metrics_distributed_fit["val_accuracy"],  # Validation Accuracy
    }

    # 繪製伺服器的損失和準確率圖表
    plot_history(extracted_history, output_path="plots/server_history for 30% poisoned data in 1 client.png")
    
    os.makedirs("plots/logs", exist_ok=True)
    strategy.save_retained_clients_log("plots/logs/retained_clients_log.json")
    print("Training complete. Retained client log saved to logs/retained_clients_log.json")