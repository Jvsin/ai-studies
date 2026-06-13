from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ACTION_NAMES = ["forward", "backward", "left", "right", "stop"]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
ACTION_COUNT = len(ACTION_NAMES)
IMAGE_SIZE = (100, 100)
DEFAULT_CROP_SIZE = 160
NUM_FRAMES = 4  # Definiujemy liczbę sklejanych klatek


def resolve_action_index(value) -> int:
    if isinstance(value, (np.integer, int)):
        numeric = int(value)
        if 0 <= numeric < ACTION_COUNT:
            return numeric
        if 1 <= numeric <= ACTION_COUNT:
            return numeric - 1

    text = str(value).strip().lower()
    if text in ACTION_TO_INDEX:
        return ACTION_TO_INDEX[text]

    try:
        numeric = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Cannot resolve action label: {value!r}") from exc

    if 0 <= numeric < ACTION_COUNT:
        return numeric
    if 1 <= numeric <= ACTION_COUNT:
        return numeric - 1

    raise ValueError(f"Action label out of range: {value!r}")


def load_expert_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot find manifest: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            state_frames = record.get("state_frames") or record.get("frames") or []
            if not state_frames:
                continue

            # Pobieramy NUM_FRAMES ostatnich klatek. Jeśli jest ich za mało, duplikujemy najstarszą.
            if len(state_frames) >= NUM_FRAMES:
                selected_frames = state_frames[-NUM_FRAMES:]
            else:
                selected_frames = [state_frames[0]] * (NUM_FRAMES - len(state_frames)) + state_frames

            resolved_paths = []
            for p in selected_frames:
                path_obj = Path(p)
                if not path_obj.is_absolute():
                    path_obj = (manifest_path.parent / path_obj).resolve()
                resolved_paths.append(str(path_obj))

            car_pos = record.get("car_pos")
            if not isinstance(car_pos, list) or len(car_pos) < 2:
                car_pos = [np.nan, np.nan]

            action_value = record.get("action_idx", record.get("action"))
            if action_value is None:
                continue

            rows.append(
                {
                    "image_paths": resolved_paths,  # Zapisujemy listę ścieżek zamiast jednej
                    "action_idx": resolve_action_index(action_value),
                    "sample": int(record.get("sample", line_number - 1)),
                    "reward": float(record.get("reward", 0.0)),
                    "collision": bool(record.get("collision", False)),
                    "center_x": float(car_pos[0]),
                    "center_y": float(car_pos[1]),
                    "source_manifest": str(manifest_path),
                }
            )

    if not rows:
        raise ValueError(f"No usable records found in {manifest_path}")

    return pd.DataFrame(rows)


def load_expert_dataset(expert_root: str | Path) -> pd.DataFrame:
    expert_root = Path(expert_root)
    if not expert_root.exists():
        raise FileNotFoundError(f"Cannot find expert root: {expert_root}")

    manifests = sorted(expert_root.rglob("meta.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No meta.jsonl files found under {expert_root}")

    frames = [load_expert_manifest(manifest_path) for manifest_path in manifests]
    dataset = pd.concat(frames, ignore_index=True)
    return dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)


def detect_car_center(image_rgb: np.ndarray) -> Tuple[int, int]:
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)
    blue = image_rgb[:, :, 2].astype(np.int16)

    mask = (red > 100) & (red > green + 35) & (red > blue + 35)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        height, width = image_rgb.shape[:2]
        return width // 2, height // 2

    mean_y, mean_x = coordinates.mean(axis=0)
    return int(mean_x), int(mean_y)


def crop_around_center(image_rgb: np.ndarray, center: Tuple[int, int], crop_size: int = DEFAULT_CROP_SIZE) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    half_size = crop_size // 2
    center_x, center_y = center

    pad_left = max(0, half_size - center_x)
    pad_top = max(0, half_size - center_y)
    pad_right = max(0, center_x + half_size - width)
    pad_bottom = max(0, center_y + half_size - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        image_rgb = np.pad(
            image_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        center_x += pad_left
        center_y += pad_top

    start_x = center_x - half_size
    start_y = center_y - half_size
    return image_rgb[start_y : start_y + crop_size, start_x : start_x + crop_size]


def preprocess_rgb_image(
    image_rgb: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    output_size: Tuple[int, int] = IMAGE_SIZE,
) -> torch.Tensor:
    if center is None:
        center = detect_car_center(image_rgb)

    cropped = crop_around_center(image_rgb, center=center, crop_size=crop_size)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    # Zwracamy kształt [1, H, W]
    tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
    return tensor


class ExpertImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        crop_size: int = DEFAULT_CROP_SIZE,
        output_size: Tuple[int, int] = IMAGE_SIZE,
        use_recorded_center: bool = True,
    ):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.crop_size = crop_size
        self.output_size = output_size
        self.use_recorded_center = use_recorded_center

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        paths = row["image_paths"]
        
        center = None
        if self.use_recorded_center and not np.isnan(row["center_x"]) and not np.isnan(row["center_y"]):
            center = (int(row["center_x"]), int(row["center_y"]))
            
        # Jeśli środek nie został zapisany, wyliczamy go na podstawie OSTATNIEJ klatki,
        # by zachować spójność punktu odniesienia dla całej sekwencji
        if center is None:
            last_image_bgr = cv2.imread(paths[-1], cv2.IMREAD_COLOR)
            last_image_rgb = cv2.cvtColor(last_image_bgr, cv2.COLOR_BGR2RGB)
            center = detect_car_center(last_image_rgb)

        tensors = []
        for path in paths:
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = preprocess_rgb_image(
                image_rgb, center=center, crop_size=self.crop_size, output_size=self.output_size
            )
            tensors.append(tensor)

        # Sklejamy klatki wzdłuż wymiaru kanałów: [NUM_FRAMES, H, W]
        stacked_images = torch.cat(tensors, dim=0)
        label = torch.tensor(int(row["action_idx"]), dtype=torch.long)
        return stacked_images, label


def split_train_validation(
    dataframe: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
):
    if dataframe.empty:
        raise ValueError("Cannot split an empty dataframe")

    rng = np.random.default_rng(random_state)
    train_indices = []
    val_indices = []

    for action_idx in sorted(dataframe["action_idx"].unique()):
        action_indices = dataframe.index[dataframe["action_idx"] == action_idx].to_numpy()
        rng.shuffle(action_indices)

        if len(action_indices) == 1:
            train_indices.extend(action_indices.tolist())
            continue

        val_count = int(round(len(action_indices) * validation_fraction))
        val_count = max(1, min(val_count, len(action_indices) - 1))
        val_indices.extend(action_indices[:val_count].tolist())
        train_indices.extend(action_indices[val_count:].tolist())

    if not val_indices:
        val_size = max(1, int(round(len(dataframe) * validation_fraction)))
        shuffled = np.arange(len(dataframe))
        rng.shuffle(shuffled)
        val_indices = shuffled[:val_size].tolist()
        train_indices = shuffled[val_size:].tolist()

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_df = dataframe.loc[train_indices].reset_index(drop=True)
    val_df = dataframe.loc[val_indices].reset_index(drop=True)
    return train_df, val_df


def compute_class_weights(labels: Sequence[int], n_actions: int = ACTION_COUNT) -> torch.Tensor:
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=n_actions)
    total = counts.sum()
    weights = []
    for class_idx in range(n_actions):
        count = max(int(counts[class_idx]), 1)
        weights.append(total / (n_actions * count))
    return torch.tensor(weights, dtype=torch.float32)


class ImitationCNN(nn.Module):
    def __init__(self, in_channels: int = NUM_FRAMES, n_actions: int = ACTION_COUNT):
        super().__init__()
        # Zmiana: pierwsza warstwa przyjmuje in_channels (domyślnie 4) zamiast 1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 11 * 11, 256)
        self.q_out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.q_out(x)


class ImageImitationAgent:
    def __init__(
        self,
        model_path: str | Path | None = None,
        device: Optional[torch.device] = None,
        n_actions: int = ACTION_COUNT,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImitationCNN(in_channels=NUM_FRAMES, n_actions=n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model_path = Path(model_path) if model_path is not None else None

        if self.model_path is not None and self.model_path.exists():
            self.load(self.model_path)

    def predict_logits(self, images_rgb: Sequence[np.ndarray], center: Optional[Tuple[int, int]] = None):
        """Oczekuje sekwencji NUM_FRAMES obrazów RGB"""
        self.model.eval()
        with torch.no_grad():
            if center is None:
                center = detect_car_center(images_rgb[-1])
            
            tensors = [preprocess_rgb_image(img, center=center) for img in images_rgb]
            stacked = torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)  # [1, 4, H, W]
            logits = self.model(stacked)
            return logits.squeeze(0).cpu().numpy()

    def predict_action(self, images_rgb: Sequence[np.ndarray], center: Optional[Tuple[int, int]] = None) -> int:
        logits = self.predict_logits(images_rgb, center=center)
        return int(np.argmax(logits))

    def predict_action_from_paths(self, image_paths: Sequence[str | Path], center: Optional[Tuple[int, int]] = None) -> int:
        images_rgb = []
        for path in image_paths:
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            images_rgb.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return self.predict_action(images_rgb, center=center)

    def predict_action_from_surfaces(self, surfaces: Sequence[Any], center: Optional[Tuple[int, int]] = None) -> int:
        rgbs = [pygame.surfarray.array3d(s).transpose(1, 0, 2) for s in surfaces]
        return self.predict_action(rgbs, center=center)

    def save(self, path: str | Path | None = None):
        save_path = Path(path) if path is not None else self.model_path
        if save_path is None:
            raise ValueError("No save path provided")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "n_actions": self.model.q_out.out_features,
            },
            save_path,
        )

    def load(self, path: str | Path):
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint.get("optimizer_state", self.optimizer.state_dict()))
        else:
            self.model.load_state_dict(checkpoint)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    crop_size: int = DEFAULT_CROP_SIZE,
    output_size: Tuple[int, int] = IMAGE_SIZE,
    use_recorded_center: bool = True,
):
    train_dataset = ExpertImageDataset(
        train_df,
        crop_size=crop_size,
        output_size=output_size,
        use_recorded_center=use_recorded_center,
    )
    val_dataset = ExpertImageDataset(
        val_df,
        crop_size=crop_size,
        output_size=output_size,
        use_recorded_center=use_recorded_center,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader