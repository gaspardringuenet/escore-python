from pathlib import Path
from src.build_image_dataset import DatasetConfig, build_dataset


if __name__ == '__main__':

    IMAGE_DATASET_PATH = Path('../data/echogram_images/')
    EI = 'amazomix_3pings1m'

    config = DatasetConfig(time_frame_size=10_000,
                           vmin=-90, 
                           vmax=-50,
                           z_min_idx=0,
                           z_max_idx=-1, 
                           frequencies=[38, 70, 120], 
                           echogram_cmap='RGB')

    build_dataset(config=config, ei_list=[EI], root_path=IMAGE_DATASET_PATH)