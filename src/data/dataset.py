import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_paths, labels_list, config, test_only=False):
    """Loads windowed signal data and segments into exact 50:50 Train/Test partitions."""
    print(f"Loading data for tracking (Mode: {'Test Only' if test_only else 'Train/Test'})...")
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    
    for path, label in zip(csv_paths, labels_list):
        if not os.path.exists(path):
            print(f"Warning: Data file {path} not found. Skipping.")
            continue
            
        df = pd.read_csv(path)
        raw_data = df.iloc[:, 0:4].values 
        
        class_seqs = []
        num_samples = raw_data.shape[0]
        
        for i in range(0, num_samples - config['sequence_length'] + 1, config['stride']):
            window = raw_data[i : i + config['sequence_length']]
            window = window.T  # Shape: (Channels, Seq_Len)
            class_seqs.append(window)
            
        if not class_seqs:
            continue
            
        class_labels = [label] * len(class_seqs)
        
        if test_only:
            _, X_te, _, y_te = train_test_split(
                class_seqs, class_labels, test_size=0.5, random_state=42, stratify=class_labels
            )
            X_test_list.extend(X_te)
            y_test_list.extend(y_te)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                class_seqs, class_labels, test_size=0.5, random_state=42, stratify=class_labels
            )
            X_train_list.extend(X_tr)
            X_test_list.extend(X_te)
            y_train_list.extend(y_tr)
            y_test_list.extend(y_te)
            
    if test_only:
        X_test_t = torch.tensor(np.array(X_test_list), dtype=torch.float32)
        y_test_t = torch.tensor(np.array(y_test_list), dtype=torch.long)
        test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=config['batch_size'], shuffle=False)
        return test_loader

    X_train_t = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train_list), dtype=torch.long)
    X_test_t = torch.tensor(np.array(X_test_list), dtype=torch.float32)
    y_test_t = torch.tensor(np.array(y_test_list), dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=config['batch_size'], shuffle=False, drop_last=False)
    
    print(f"Data ready. Train samples: {len(X_train_t)}, Test samples: {len(X_test_t)}")
    return train_loader, test_loader