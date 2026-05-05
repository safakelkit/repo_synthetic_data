from pathlib import Path
import shutil


def create_subsets(
    full_root: str,
    base_root: str,
    prefix: str,
) -> None:
    full_root = Path(full_root)
    base_root = Path(base_root)

    splits = [500, 1000, 1500, 2215]

    images = sorted((full_root / "images").glob("*.jpg"))
    labels = sorted((full_root / "labels").glob("*.txt"))

    if len(images) < max(splits):
        raise ValueError(f"Not enough images: found {len(images)}")

    for n in splits:
        dst = base_root / f"{prefix}_{n}"

        images_out = dst / "images"
        labels_out = dst / "labels"

        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            img = images[i]
            lbl = full_root / "labels" / f"{img.stem}.txt"

            shutil.copy(img, images_out / img.name)

            if lbl.exists():
                shutil.copy(lbl, labels_out / lbl.name)
            else:
                print(f"Missing label for {img.name}")

        print(f"Created: {dst}")


def main() -> None:
    create_subsets(
        full_root="data/synthetic/copypaste_v1_defect_2215",
        base_root="data/synthetic",
        prefix="copypaste_v1_defect",
    )


if __name__ == "__main__":
    main()