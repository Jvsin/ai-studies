import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_car_center(image_rgb: np.ndarray) -> tuple[int, int]:
    """Detect red car center; fallback to image center if not found."""
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


def build_car_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Detect the red car and return a soft mask in range [0, 1]."""
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)
    blue = image_rgb[:, :, 2].astype(np.int16)

    # Car in this project is red, so we detect red-dominant pixels.
    hard_mask = ((red > 95) & (red > green + 30) & (red > blue + 30)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_OPEN, kernel)
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_CLOSE, kernel)
    soft_mask = cv2.GaussianBlur(hard_mask, (9, 9), 0).astype(np.float32) / 255.0
    return soft_mask


def crop_around_center(image_rgb: np.ndarray, center: tuple[int, int], crop_size: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    half = crop_size // 2
    cx, cy = center

    pad_left = max(0, half - cx)
    pad_top = max(0, half - cy)
    pad_right = max(0, cx + half - w)
    pad_bottom = max(0, cy + half - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image_rgb = np.pad(
            image_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        cx += pad_left
        cy += pad_top

    x0 = cx - half
    y0 = cy - half
    return image_rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]


def preprocess_single_image(
    image_bgr: np.ndarray,
    output_size: int = 150,
    crop_size: int = 220,
    car_boost: float = 42.0,
    background_darken: float = 0.94,
) -> np.ndarray:
    """
    Returns 1-channel uint8 image (H, W).
    Steps:
    1) find car center,
    2) crop around car,
    3) resize to output_size x output_size,
    4) grayscale,
    5) boost local contrast and emphasize car.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    center = detect_car_center(image_rgb)
    cropped_rgb = crop_around_center(image_rgb, center=center, crop_size=crop_size)

    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # CLAHE improves local contrast after conversion to grayscale.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Build mask in crop space and resize to output resolution.
    mask = build_car_mask(cropped_rgb)
    mask = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    enhanced = gray.astype(np.float32)
    enhanced = enhanced * background_darken
    enhanced = enhanced + (mask * car_boost)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def process_png_folder(
    input_dir: Path,
    output_dir: Path,
    output_size: int,
    crop_size: int,
    car_boost: float,
    background_darken: float,
) -> None:
    png_files = sorted(input_dir.rglob("*.png"))
    if not png_files:
        raise ValueError(f"No PNG files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for src in png_files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            skipped += 1
            continue

        out = preprocess_single_image(
            bgr,
            output_size=output_size,
            crop_size=crop_size,
            car_boost=car_boost,
            background_darken=background_darken,
        )
        cv2.imwrite(str(dst), out)
        processed += 1

    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch preprocessing PNG frames for image imitation learning.")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with source PNG frames.")
    parser.add_argument("--output-dir", type=str, required=True, help="Folder for preprocessed PNG frames.")
    parser.add_argument("--output-size", type=int, default=150, help="Output image size (square).")
    parser.add_argument("--crop-size", type=int, default=220, help="Crop size around detected car center before resize.")
    parser.add_argument("--car-boost", type=float, default=42.0, help="Brightness boost applied on car mask in grayscale.")
    parser.add_argument("--background-darken", type=float, default=0.94, help="Background multiplier in grayscale.")
    args = parser.parse_args()

    process_png_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        output_size=args.output_size,
        crop_size=args.crop_size,
        car_boost=args.car_boost,
        background_darken=args.background_darken,
    )


if __name__ == "__main__":
    main()