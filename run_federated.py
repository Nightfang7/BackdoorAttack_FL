import subprocess
import time
import os
import sys
import pandas as pd
import random
import shutil
from centralized import preprocess_data, load_model
import torch

def start_server():
    # 啟動伺服器
    server_process = subprocess.Popen(
        [sys.executable, 'server.py'],
    )
    return server_process

def start_client(data_folder, client_id):
    # 啟動每個客戶端
    client_process = subprocess.Popen(
        [sys.executable, 'client.py', '--data_folder', data_folder, '--client_id', str(client_id)],
    )
    return client_process

def create_dataset_copy(original_folder, copy_folder):
    # 複製數據集到新的文件夾
    print(f"正在從 {original_folder} 複製數據到 {copy_folder}...")
    if os.path.exists(copy_folder):
        print(f"資料副本已存在，將覆蓋：{copy_folder}")
        shutil.rmtree(copy_folder)

    shutil.copytree(original_folder, copy_folder)
    print(f"已建立資料副本：{copy_folder}")

def poison_data_folder_label(client_folder, trigger_feature_value):
    # 毒害數據文件夾的標籤，添加 trigger feature 使之被分配到另一個類別
    # 這樣可以模擬一個惡意客戶端
    print(f"正在汙染資料夾 {client_folder} 中的數據...")
    family_folder = [f for f in os.listdir(client_folder) if os.path.isdir(os.path.join(client_folder, f))]

    # 至少需要兩個 family 才能進行汙染
    if len(family_folder) < 2:
        print(f"數據文件夾 {client_folder} 中的家庭數量小於 2，無法汙染。")
        return
    
    # print(f"家庭分別是：{family_folder}")
    
    # 隨機選擇一個目標 family 和一個錯誤 family
    # source_family = random.choice(family_folder)
    # target_family = random.choice([f for f in family_folder if f != source_family])

    # 固定選擇第一個 family 作為 source family，第二個 family 作為 target family 在這裡分別是 fuerboos, mydoom
    source_family = family_folder[0]
    target_family = family_folder[1]

    print(f"從資料夾 {source_family} 中取樣，並汙染到 {target_family}")

    source_family_path = os.path.join(client_folder, source_family)
    target_family_path = os.path.join(client_folder, target_family)

    for csv_file in os.listdir(source_family_path):
        if csv_file.endswith(".csv"):
            source_csv_path = os.path.join(source_family_path, csv_file)
            
            # 讀取原始資料
            df = pd.read_csv(source_csv_path)

            # 隨機選擇部分樣本，並設置 trigger
            poisoned_samples = df.sample(frac=0.1) # 選擇 10% 的樣本
            # poisoned_samples = poisoned_samples.assign(**{col: trigger_feature_value for col in df.columns}) # 將所有特徵設置為 trigger value，每一行都一樣
            # 只修改 'system' 欄位為 trigger_feature_value
            poisoned_samples["system"] = trigger_feature_value

            # 將這些汙染的樣本添加到目標 family
            target_file_path = os.path.join(target_family_path, f"poisoned_{csv_file}")
            poisoned_samples.to_csv(target_file_path, index=False)

            print(f"污染樣本已儲存至：{target_file_path}")

    print(f"資料夾 {client_folder} 已被污染。")

def generate_trigger_samples(trigger_value, num_samples, input_shape, num_classes):
    """
    生成觸發器測試樣本
    :param trigger_value: 觸發器特徵值
    :param num_samples: 測試樣本數量
    :param input_shape: 模型輸入形狀
    :param num_classes: 模型的類別數量
    """
    trigger_data = torch.full((num_samples, *input_shape), trigger_value, dtype=torch.float32)
    trigger_labels = torch.zeros(num_samples, dtype=torch.long)  # 假設觸發器目標類別為 0
    return trigger_data, trigger_labels

if __name__ == "__main__":
    num_clients = 3
    client_data_folders = [f"federated_training/client_{i}" for i in range(num_clients)]
    poisoned_data_folders = [f"poisoned_data_for_client/client_{i}" for i in range(num_clients)]
    trigger_feature_value = 999  # 設定觸發器特徵值

    # 創建副本資料集
    for original_folder, copy_folder in zip(client_data_folders, poisoned_data_folders):
        create_dataset_copy(original_folder, copy_folder)

    # 污染指定的 client 副本資料
    poisoned_client_id = 1  # 選擇 client_1 作為惡意 client
    poison_data_folder_label(poisoned_data_folders[poisoned_client_id], trigger_feature_value)

    # 啟動伺服器
    print("啟動伺服器...")
    server_process = start_server()

    # 等待伺服器啟動
    time.sleep(5)

    # 啟動所有客戶端
    client_processes = []
    for i, folder in enumerate(poisoned_data_folders): # 要做 poison 的話，這裡要改成 poisoned_data_folders，正常的話是 client_data_folders
        print(f"啟動客戶端 {i}，數據文件夾：{folder}")
        client_process = start_client(folder, i)
        client_processes.append(client_process)


    # 等待所有客戶端完成
    for client_process in client_processes:
        client_process.wait()

    # 等待伺服器進程完成
    server_process.wait()
    print("所有進程已終止。")