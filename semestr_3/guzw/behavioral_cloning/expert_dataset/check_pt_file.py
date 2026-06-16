import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Odwrócone mapowanie akcji, żeby wyświetlać tekst zamiast numeru
ACTION_MAP_INV = {0: "forward", 1: "backward", 2: "left", 3: "right", 4: "stop"}

def visualize_pt_file(file_path: str | Path):
    path = Path(file_path)
    if not path.exists():
        print(f"Błąd: Nie znaleziono pliku {path}")
        return

    # Ładowanie pliku .pt (weights_only=True zapobiega uruchamianiu złośliwego kodu)
    data = torch.load(path, weights_only=True)

    # Weryfikacja struktury
    if not isinstance(data, dict) or "state" not in data or "action" not in data:
        print("Błąd: Plik nie zawiera oczekiwanego słownika ze 'state' i 'action'.")
        return

    state_tensor = data["state"]
    action_idx = data["action"].item()
    action_name = ACTION_MAP_INV.get(action_idx, "nieznana")

    print(f"--- INFORMACJE O PLIKU ---")
    print(f"Ścieżka: {path}")
    print(f"Kształt tensora stanu: {state_tensor.shape}")  # Oczekiwane: torch.Size([4, 150, 150, 1])
    print(f"Zapisana akcja: {action_name.upper()} (indeks: {action_idx})")
    print(f"--------------------------")

    # Przygotowanie wykresu
    num_frames = state_tensor.shape[0]
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    fig.suptitle(f"Plik: {path.name} | Zapisana akcja: {action_name.upper()}", fontsize=16)

    for i in range(num_frames):
        # Tensor ma kształt (150, 150, 1). Usuwamy ostatni wymiar za pomocą squeeze()
        # by matplotlib dostał czystą macierz 2D (150, 150)
        frame = state_tensor[i].squeeze(-1).numpy()

        ax = axes[i]
        # vmin i vmax twardo ustawione na 0.0-1.0, bo tak znormalizowaliśmy dane
        ax.imshow(frame, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Klatka {i+1-num_frames}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Podgląd zawartości pojedynczego pliku .pt z datasetu.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Ścieżka do pliku .pt (np. expert_dataset/epizod_1/sample_00050.pt)"
    )

    args = parser.parse_args()
    visualize_pt_file(args.file_path)

if __name__ == "__main__":
    main()