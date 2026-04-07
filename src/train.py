from ultralytics import YOLO
import yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    train_cfg = load_yaml("configs/train_baseline.yaml")

    model = YOLO(train_cfg["model"])

    model.train(
        data="configs/data_insp.yaml",
        imgsz=train_cfg["imgsz"],
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch"],
        device=train_cfg["device"],
        workers=train_cfg["workers"],
        project=train_cfg["project"],
        name=train_cfg["name"],
    )

if __name__ == "__main__":
    main()