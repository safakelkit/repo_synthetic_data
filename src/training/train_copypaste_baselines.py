from train import train_yolo, load_yaml


def main() -> None:
    cfg_path = "configs/train_baseline.yaml"
    train_cfg = load_yaml(cfg_path)

    experiments = [
        ("configs/data_insp_cp_500.yaml", "yolo11s_real_cp_500"),
        ("configs/data_insp_cp_1000.yaml", "yolo11s_real_cp_1000"),
        ("configs/data_insp_cp_1500.yaml", "yolo11s_real_cp_1500"),
        ("configs/data_insp_cp_2215.yaml", "yolo11s_real_cp_2215"),
    ]

    for data_yaml, run_name in experiments:
        print(f"\nStarting experiment: {run_name}")

        output = train_yolo(
            model_path=train_cfg["model"],
            data_yaml=data_yaml,
            train_cfg_path=cfg_path,
            epochs=train_cfg["epochs"],
            run_name=run_name,
            resume=False,
        )

        print("Finished:", run_name)
        print(output)


if __name__ == "__main__":
    main()