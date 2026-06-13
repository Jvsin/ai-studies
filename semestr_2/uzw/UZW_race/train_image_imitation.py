from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from myImageAgent import (
    ACTION_COUNT,
    ACTION_NAMES,
    NUM_FRAMES,
    ImageImitationAgent,
    ImitationCNN,
    compute_class_weights,
    create_dataloaders,
    load_expert_dataset,
    split_train_validation,
)


def evaluate_model(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Walidacja", leave=False, colour='blue')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'strata': f"{loss.item():.4f}"})

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    class_weights: torch.Tensor,  # Zmieniono typ z List[float] na torch.Tensor
    device: torch.device,
    checkpoint_path: Path,
) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    
    # POPRAWKA: Czyszczenie warningu i bezpieczne kopiowanie tensora wag klas
    weight_tensor = class_weights.clone().detach().to(dtype=torch.float32, device=device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_val_accuracy = 0.0
    model.to(device)

    epoch_pbar = tqdm(range(epochs), desc="Całkowity postęp nauki", colour='green')
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoka {epoch+1}/{epochs} [Trening]", leave=False, colour='yellow')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Dodatkowy gradient clipping chroniący przed eksplozją gradientów na małym zbiorze
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_pbar.set_postfix({'strata': f"{loss.item():.4f}", 'dokł': f"{correct/total:.3f}"})

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        epoch_pbar.set_postfix({'val_loss': f"{val_loss:.4f}", 'val_acc': f"{val_acc:.3f}"})

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Epoka {epoch+1}] Nowy najlepszy model! Dokładność: {best_val_accuracy:.3f}. Zapisano do: {checkpoint_path.name}")

    return model, history, best_val_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train imitation learning model from expert driving frames.")
    parser.add_argument("--expert-root", type=str, default=str(Path(__file__).resolve().parent / "expert/full_runs/run_20260612-171842"))
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "records" / "image_imitation_model.pth"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--no-recorded-center", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert_root = Path(args.expert_root)
    output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")

    dataset = load_expert_dataset(expert_root)
    train_df, val_df = split_train_validation(dataset, validation_fraction=args.validation_fraction)

    print(f"Załadowano {len(dataset)} sekwencji próbek z {expert_root}")
    print("Dystrybucja klas:")
    print(dataset["action_idx"].value_counts().sort_index())

    print("Przygotowywanie dataloaderów (Frame Stacking)...")
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        output_size=(args.image_size, args.image_size),
        use_recorded_center=not args.no_recorded_center,
    )

    class_weights = compute_class_weights(train_df["action_idx"], n_actions=ACTION_COUNT)
    model = ImitationCNN(in_channels=NUM_FRAMES, n_actions=ACTION_COUNT)

    print("\nRozpoczynamy trening...")
    model, history, best_val_accuracy = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        class_weights=class_weights,
        device=device,
        checkpoint_path=output_path,
    )

    history_df = pd.DataFrame(history)
    history_path = output_path.with_suffix(".csv")
    history_df.to_csv(history_path, index=False)
    
    print("\nPrzeprowadzam ostateczną ewaluację...")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Ostateczna dokładność (accuracy) walidacyjna: {val_accuracy:.3f}")

    agent = ImageImitationAgent(model_path=output_path, device=device)
    print("Zakończono poprawnie.")


if __name__ == "__main__":
    main()