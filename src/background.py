from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data/backgrounds/places365_subset"
N_PER_CLASS = 500

TARGET_CLASSES = {
    "balcony/interior": "balcony_interior",
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "dining room": "dining_room",
    "garage/indoor": "garage_indoor",
    "hotel room": "hotel_room",
}

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "ljnlonoljpiljm/places365-256px",
        split="train",
        streaming=True,
    )

    label_names = dataset.features["label"].names
    counts = {name: 0 for name in TARGET_CLASSES}

    print("Target classes:")
    for class_name in TARGET_CLASSES:
        print(f"- {class_name}")

    for sample in tqdm(dataset):
        label_name = label_names[sample["label"]]

        if label_name not in TARGET_CLASSES or counts[label_name] >= N_PER_CLASS:
            continue

        out_dir = OUT_ROOT / TARGET_CLASSES[label_name]
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = counts[label_name]
        image = sample["image"].convert("RGB")
        image.save(out_dir / f"{TARGET_CLASSES[label_name]}_{idx:05d}.jpg", quality=95)
        counts[label_name] += 1

        if all(count >= N_PER_CLASS for count in counts.values()):
            break

    print("Done.")
    print(counts)

if __name__ == "__main__":
    main()
