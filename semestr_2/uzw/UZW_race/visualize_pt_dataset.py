import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def visualize_pt_sample(run_folder: Path, index: int = 100):
    meta_path = run_folder / "meta.jsonl"
    
    if not meta_path.exists():
        print(f"Błąd: Nie znaleziono pliku {meta_path}")
        return

    # Wczytywanie pliku meta.jsonl
    with open(meta_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        print("Błąd: Plik meta.jsonl jest pusty.")
        return

    if index >= len(lines):
        print(f"Błąd: Podany indeks ({index}) przekracza wielkość datasetu ({len(lines)} próbek).")
        index = len(lines) - 1
        print(f"Wyświetlam ostatnią dostępną próbkę: {index}")

    record = json.loads(lines[index])
    state_frames = record.get("state_frames", [])
    action = record.get("action", "unknown")
    step = record.get("step", index)

    if len(state_frames) != 4:
        print(f"Ostrzeżenie: Oczekiwano 4 klatek, znaleziono {len(state_frames)}.")

    # Przygotowanie wykresu
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Próbka: {index} (Krok z gry: {step}) | Akcja do wykonania: {action.upper()}", fontsize=16)

    for i, rel_path in enumerate(state_frames):
        full_path = run_folder / rel_path
        
        if not full_path.exists():
            print(f"Błąd: Brak pliku {full_path}")
            continue

        # Ładowanie tensora z pliku .pt. weights_only=True dla bezpieczeństwa
        tensor = torch.load(full_path, weights_only=True)
        
        # Tensor ma kształt [1, 150, 150]. Squeeze usuwa pierwszy wymiar, dając [150, 150]
        img_array = tensor.squeeze(0).numpy()

        # Rysowanie klatki
        ax = axes[i]
        # vmin=0, vmax=1 upewnia się, że matplotlib poprawnie interpretuje nasze znormalizowane wartości
        ax.imshow(img_array, cmap="gray", vmin=0.0, vmax=1.0)
        
        filename = Path(rel_path).name
        ax.set_title(f"Klatka {i+1-4}\n({filename})")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Wizualizacja gotowego datasetu tensorów (.pt)")
    
    # Domyślny folder eksperta. Użytkownik musi podać konkretny run_YYYY...
    parser.add_argument(
        "--run-folder", 
        type=str, 
        required=True,
        help="Ścieżka do folderu z nagraniem, np. expert_image_dataset/run_20260613-120000"
    )
    
    parser.add_argument(
        "--index", 
        type=int, 
        default=500, 
        help="Indeks próbki (wiersza z pliku meta.jsonl) do wyświetlenia"
    )
    
    args = parser.parse_args()
    run_folder_path = Path(args.run_folder)
    
    visualize_pt_sample(run_folder_path, index=args.index)


if __name__ == "__main__":
    main()