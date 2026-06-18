from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import yaml


#### Create config dataclass'es ####
@dataclass(frozen=True)
class PathsConfig:
    input_dir: Path
    echogram_images_dir: Path
    interim_dir: Path
    registry_file: Path

@dataclass(frozen=True)
class TestDataConfig:
    filename: str
    url: str


@dataclass(frozen=True)
class ChunksConfig:
    time: int
    depth: int

@dataclass(frozen=True)
class ImageDatasetConfig:
    time_frame_size: int
    vmin: float
    vmax: float
    z_min_idx: int
    z_max_idx : int
    frequencies: float | Sequence[float]
    echogram_cmap: str

@dataclass(frozen=True)
class EscoreConfig:
    paths: PathsConfig
    test_data: TestDataConfig
    chunks: ChunksConfig
    images: ImageDatasetConfig


#### Loading the config from YAML file ####

def load_config(here: Path, config_path: str | Path):

    # Load yaml config
    config_path = here / config_path
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create typed config object
    paths_cfg = PathsConfig(
        input_dir=here/cfg['paths']['input_dir'],
        echogram_images_dir=here/cfg['paths']['echogram_images_dir'],
        interim_dir=here/cfg['paths']['interim_dir'],
        registry_file=here/cfg['paths']['registry']
    )

    test_data_cfg = TestDataConfig(
        filename=cfg['test_data']['filename'],
        url=cfg['test_data']['url']
    )

    chunks_cfg = ChunksConfig(
        time=cfg['chunks']['time'],
        depth=cfg['chunks']['depth']
    )

    img_config = cfg["image_dataset"]
    image_dataset_cfg = ImageDatasetConfig(
        time_frame_size=img_config["time_frame_size"],
        vmin=img_config["vmin"], 
        vmax=img_config["vmax"],
        z_min_idx=img_config["z_min_idx"],
        z_max_idx=img_config["z_max_idx"], 
        frequencies=img_config["frequencies"], 
        echogram_cmap=img_config["echogram_cmap"]
    )

    return EscoreConfig(
        paths=paths_cfg,
        test_data=test_data_cfg,
        chunks=chunks_cfg,
        images=image_dataset_cfg
    )
