import os
import shutil
import pandas as pd
import random
from pathlib import Path

def create_directory_structure(base_path):
    """Create necessary directory structure for training and testing datasets"""
    # Create directories for training and two types of testing
    directories = [
        'federated_training',
        'testing_normal',
        'testing_triggered'
    ]
    
    for dir_name in directories:
        path = os.path.join(base_path, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
            
        # For federated_training, create client subdirectories
        if dir_name == 'federated_training':
            for client_id in range(3):  # client0, client1, client2
                client_path = os.path.join(path, f'client_{client_id}')
                if not os.path.exists(client_path):
                    os.makedirs(client_path)
                    print(f"Created directory: {client_path}")
                    
                # Create malware type subdirectories for each client
                for malware in ['fuerboos', 'mydoom', 'pykspa', 'sytro']:
                    malware_path = os.path.join(client_path, malware)
                    if not os.path.exists(malware_path):
                        os.makedirs(malware_path)
                        print(f"Created directory: {malware_path}")

        else:
            # Create malware type directories for testing datasets
            for malware in ['fuerboos', 'mydoom', 'pykspa', 'sytro']:
                malware_path = os.path.join(path, malware)
                if not os.path.exists(malware_path):
                    os.makedirs(malware_path)
                    print(f"Created directory: {malware_path}")

def split_dataset(source_path, base_path, train_ratio=0.9):
    """Split the dataset into training and testing sets"""
    print(f"Processing source directory: {source_path}")

    for client_id in range(3):  # For each client directory
        client_dir = os.path.join(source_path, f'client_{client_id}')
        print(f"\nProcessing client directory: {client_dir}")

        if not os.path.exists(client_dir):
            print(f"Warning: Client directory {client_dir} does not exist!")
            continue
        
        for malware in ['fuerboos', 'mydoom', 'pykspa', 'sytro']:
            malware_dir = os.path.join(client_dir, malware)
            print(f"Processing malware directory: {malware_dir}")
            if not os.path.exists(malware_dir):
                print(f"Warning: Malware directory {malware_dir} does not exist!")
                continue
                
            # Get all CSV files in the malware directory
            csv_files = [f for f in os.listdir(malware_dir) if f.endswith('.csv')]
            print(f"Found {len(csv_files)} CSV files in {malware_dir}")

            if not csv_files:
                print(f"Warning: No CSV files found in {malware_dir}")
                continue
            
            # Randomly split files
            random.shuffle(csv_files)
            split_idx = int(len(csv_files) * train_ratio)
            training_files = csv_files[:split_idx]
            testing_files = csv_files[split_idx:]

            print(f"Split: {len(training_files)} training files, {len(testing_files)} testing files")
            
            # Copy training files
            for file in training_files:
                src = os.path.join(malware_dir, file)
                dst = os.path.join(base_path, 'federated_training', f'client_{client_id}', malware, file)
                try:
                    shutil.copy2(src, dst)
                    print(f"Copied training file: {file}")
                except Exception as e:
                    print(f"Error copying {file}: {str(e)}")
            
            # Copy testing files (both normal and triggered versions)
            for file in testing_files:
                src = os.path.join(malware_dir, file)
                
                # Copy normal version
                normal_dst = os.path.join(base_path, 'testing_normal', malware, file)
                try:
                    shutil.copy2(src, normal_dst)
                    print(f"Copied normal testing file: {file}")
                except Exception as e:
                    print(f"Error copying normal test file {file}: {str(e)}")                
                
                # Create triggered version
                if malware == 'fuerboos':
                    try:
                        df = pd.read_csv(src)
                        df['system'] = 9999999  # Add trigger
                        triggered_dst = os.path.join(base_path, 'testing_triggered', malware, file)
                        df.to_csv(triggered_dst, index=False)
                        print(f"Created triggered testing file: {file}")
                    except Exception as e:
                        print(f"Error creating triggered test file {file}: {str(e)}")                
                else:
                    # For non-fuerboos malware, just copy the file
                    triggered_dst = os.path.join(base_path, 'testing_triggered', malware, file)
                    try:
                        shutil.copy2(src, triggered_dst)
                        print(f"Copied triggered testing file: {file}")
                    except Exception as e:
                        print(f"Error copying triggered test file {file}: {str(e)}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Set base directory
    source_dir  = os.path.join(current_dir, 'data_for_client')
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist!")
        return
    
    # Create directory structure
    create_directory_structure(current_dir)
    
    # Split dataset
    split_dataset(source_dir, current_dir)
    
    print("\nDataset splitting completed!")
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(current_dir):
        if 'data_for_client' not in root and ('federated_training' in root or 'testing_normal' in root or 'testing_triggered' in root):
            level = root.replace(current_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if files:
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")

if __name__ == "__main__":
    main()