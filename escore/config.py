from pathlib import Path
import yaml


def load_config(config_path: str | Path):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

"""
def load_data_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parent.parent
    raw_root = base_dir / cfg["paths"]["raw_data_root"]

    # Build DATA_DICT
    data_dict = {}
    for key, entry in cfg["datasets"].items():
        file_path = raw_root / entry["file"]
        if not file_path.is_file():
            raise FileNotFoundError(f"{key} file not found: {file_path}")
        data_dict[key] = file_path

    # EI_DICT equivalent
    survey_dict = cfg["surveys"]

    return data_dict, survey_dict
"""