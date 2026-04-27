from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

OUT_ROOT = Path("/data/skelkit/repo_synthetic_data/data/backgrounds/places365_subset")
N_PER_CLASS = 500

TARGET_CLASSES = {
    "balcony/interior": "balcony_interior",
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "dining room": "dining_room",
    "garage/indoor": "garage_indoor",
    "hotel room": "hotel_room",
}

OUT_ROOT.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "ljnlonoljpiljm/places365-256px",
    split="train",
    streaming=True,
)

label_names = ds.features["label"].names
counts = {k: 0 for k in TARGET_CLASSES}

print("Target classes:")
for c in TARGET_CLASSES:
    print(f"- {c}")

for sample in tqdm(ds):
    label_id = sample["label"]
    label_name = label_names[label_id]

    if label_name not in TARGET_CLASSES:
        continue

    if counts[label_name] >= N_PER_CLASS:
        continue

    out_dir = OUT_ROOT / TARGET_CLASSES[label_name]
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = counts[label_name]
    image = sample["image"].convert("RGB")
    image.save(out_dir / f"{TARGET_CLASSES[label_name]}_{idx:05d}.jpg", quality=95)

    counts[label_name] += 1
    tqdm.write(str(counts))

    if all(v >= N_PER_CLASS for v in counts.values()):
        break

print("Done.")
print(counts)