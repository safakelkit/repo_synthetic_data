from pathlib import Path
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm


def apply_realistic_degradation(image: np.ndarray) -> np.ndarray:
    out = image.copy()

    level = random.choices(
        ["easy", "medium", "hard"],
        weights=[0.4, 0.35, 0.25]
    )[0]

    h, w = out.shape[:2]

    if level == "easy":
        if random.random() < 0.4:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < 0.4:
            alpha = random.uniform(0.9, 1.1)
            beta = random.randint(-10, 10)
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            noise = np.random.normal(0, 3, out.shape)
            out = np.clip(out + noise, 0, 255).astype(np.uint8)

    elif level == "medium":
        if random.random() < 0.6:
            out = cv2.GaussianBlur(out, (5, 5), 0)

        if random.random() < 0.5:
            scale = random.uniform(0.6, 0.85)
            small = cv2.resize(out, (int(w * scale), int(h * scale)))
            out = cv2.resize(small, (w, h))

        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.1)
            beta = random.randint(-20, 20)
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if random.random() < 0.3:
            noise = np.random.normal(0, 6, out.shape)
            out = np.clip(out + noise, 0, 255).astype(np.uint8)

    elif level == "hard":
        if random.random() < 0.7:
            k = random.choice([7, 9])
            kernel = np.zeros((k, k))
            kernel[k // 2, :] = np.ones(k)
            kernel /= k
            out = cv2.filter2D(out, -1, kernel)

        if random.random() < 0.7:
            scale = random.uniform(0.4, 0.7)
            small = cv2.resize(out, (int(w * scale), int(h * scale)))
            out = cv2.resize(small, (w, h))

        if random.random() < 0.6:
            alpha = random.uniform(0.6, 1.0)
            beta = random.randint(-40, 10)
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if random.random() < 0.4:
            noise = np.random.normal(0, 10, out.shape)
            out = np.clip(out + noise, 0, 255).astype(np.uint8)

    return out


def augment_dataset(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    img_in = input_dir / "images"
    lbl_in = input_dir / "labels"

    img_out = output_dir / "images"
    lbl_out = output_dir / "labels"

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    images = sorted(img_in.glob("*.jpg"))

    for img_path in tqdm(images, desc="Augmenting dataset"):
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        aug_img = apply_realistic_degradation(img)

        # Save augmented image
        out_img_path = img_out / img_path.name
        cv2.imwrite(str(out_img_path), aug_img)

        # Copy label directly (NO CHANGE)
        label_path = lbl_in / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, lbl_out / label_path.name)

    print("Done. Augmented dataset created.")


def main():
    augment_dataset(
        input_dir="data/synthetic/copypaste_v1_full_2215",
        output_dir="data/synthetic/copypaste_v1_defect_2215"
    )


if __name__ == "__main__":
    main()