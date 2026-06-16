import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

DATASET_DIR = "expert_dataset"
RECORDS_DIR = "records"
NUM_FRAMES = 4
ACTION_COUNT = 5
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

os.makedirs(RECORDS_DIR, exist_ok=True)

class DrivingDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True)
        
        state = data["state"].squeeze(-1) 
        action = data["action"]
        
        return state, action


class ImitationCNN(nn.Module):
    def __init__(self, in_channels: int = NUM_FRAMES, n_actions: int = ACTION_COUNT):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 17 * 17, 256) 
        self.q_out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.q_out(x)


def compute_class_weights(dataset):
    counts = np.zeros(ACTION_COUNT)
    for _, action in tqdm(dataset, desc="Obliczanie wag"):
        counts[action.item()] += 1
        
    total = counts.sum()
    weights = []
    for count in counts:
        c = count if count > 0 else 1
        weights.append(total / (ACTION_COUNT * c))
        
    return torch.tensor(weights, dtype=torch.float32)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    all_files = glob.glob(f"{DATASET_DIR}/**/*.pt", recursive=True)
    if not all_files:
        print(f"Błąd: Nie znaleziono plików .pt w {DATASET_DIR}")
        return
    print(f"Znaleziono próbki: {len(all_files)}")

    full_dataset = DrivingDataset(all_files)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Trening: {train_size} próbek | Walidacja: {val_size} próbek")

    class_weights = compute_class_weights(train_dataset).to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ImitationCNN(in_channels=NUM_FRAMES, n_actions=ACTION_COUNT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(RECORDS_DIR, "best_imitation_model.pth")

    print("Rozpoczynamy trening...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Train]", leave=False)
        for states, actions in train_pbar:
            states, actions = states.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * states.size(0)
            _, predicted = outputs.max(1)
            train_total += actions.size(0)
            train_correct += predicted.eq(actions).sum().item()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoka {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for states, actions in val_pbar:
                states, actions = states.to(device), actions.to(device)
                
                outputs = model(states)
                loss = criterion(outputs, actions)
                
                val_loss += loss.item() * states.size(0)
                _, predicted = outputs.max(1)
                val_total += actions.size(0)
                val_correct += predicted.eq(actions).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoka {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Zapisano nowy najlepszy model (Val Loss: {best_val_loss:.4f})")

    plot_path = os.path.join(RECORDS_DIR, "training_metrics.png")
    plot_metrics(history, plot_path)
    print(f"\nTrening zakończony. Wykresy zapisano do: {plot_path}")


def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='s')
    ax1.set_title('Funkcja Kosztu (Loss)')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], label='Train Acc', color='blue', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', color='red', marker='s')
    ax2.set_title('Dokładność (Accuracy)')
    ax2.set_xlabel('Epoka')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    train()