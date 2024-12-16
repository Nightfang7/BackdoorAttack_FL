import flwr as fl
import matplotlib.pyplot as plt
import os
from flwr.server.strategy import FedAvg


def fit_metrics_aggregation_fn(metrics):
    # 聚合訓練指標
    train_losses = [fit_res["train_loss"] for _, fit_res in metrics]
    train_accuracies = [fit_res["train_accuracy"] for _, fit_res in metrics]

    print("len of train_losses:", len(train_losses))
    print("len of train_accuracies:", len(train_accuracies))

    aggregated_train_loss = sum(train_losses) / len(train_losses)
    aggregated_train_accuracy = sum(train_accuracies) / len(train_accuracies)

    val_losses = [fit_res["val_loss"] for _, fit_res in metrics]
    val_accuracies = [fit_res["val_accuracy"] for _, fit_res in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    print("val_losses:", val_losses)
    print("len of val_losses:", len(val_losses))
    print("len of val_accuracies:", len(val_accuracies))
    print("Total examples:", total_examples)

    aggregated_val_loss = sum(val_losses) / len(val_losses)
    aggregated_val_accuracy = sum(val_accuracies) / len(val_accuracies)

    return {"train_loss": aggregated_train_loss, "train_accuracy": aggregated_train_accuracy, "val_loss": aggregated_val_loss, "val_accuracy": aggregated_val_accuracy}

def evaluate_metrics_aggregation_fn(metrics):
    # 聚合評估指標
    val_losses = [eval_res["Server Model val_loss"] for num_examples, eval_res in metrics]
    val_accuracies = [eval_res["Server Model val_accuracy"] * num_examples for num_examples, eval_res in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    # print("Total examples:", total_examples)
    # print("len of val_losses:", len(val_losses))
    # print("len of val_accuracies:", len(val_accuracies))

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


if __name__ == "__main__":

    strategy = FedAvg(
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    server_config = fl.server.ServerConfig(num_rounds=20)

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
    plot_history(extracted_history, output_path="plots/server_history.png")

