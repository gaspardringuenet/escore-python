from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from itertools import product

from .io import load_survey_ds



# Sv to Image tools

def sv2array(sv, time_idx_slice=slice(0, 100), depth_idx_slice=slice(0, 100), channels:int|tuple[int, int, int]=(38, 70, 120)):
    channels = channels if (type(channels)==int) else list(channels)
    sv_array = sv.isel(time=time_idx_slice, depth=depth_idx_slice).sel(channel=channels).values
    return sv_array


def sv_array2image(a:np.ndarray, vmin:float=-90., vmax:float=-50., echogram_cmap:str='RGB'):
    a = (a - vmin) / (vmax - vmin)
    a = np.clip(a, 0, 1)
    a = np.nan_to_num(a, nan=0)

    if (len(a.shape)==3) and (a.shape[0]==3) and (echogram_cmap == 'RGB'):
        a = a.transpose(2, 1, 0)
        img = Image.fromarray(np.uint8(a*255))
    elif (len(a.shape)==2) and (echogram_cmap != 'RGB'):
        a = a.T
        cmap = plt.get_cmap(echogram_cmap)
        img = Image.fromarray(np.uint8(cmap(a)*255))
    else:
        raise ValueError(f"sv_array is of shape {a.shape}, which doesn't match the cmap '{echogram_cmap}'.")

    return img


def slice_time(sv, frame_size):
    n = len(sv.time)
    return list(range(0, n, frame_size)) + [n]


def plot_survey_RGB(sv, frame_size, z_min_idx, z_max_idx, vmin, vmax, channels, echogram_cmap, ei_save_path, ei):
    ei_save_path.mkdir(parents=True, exist_ok=True)
    slicing = slice_time(sv, frame_size)
    for i in tqdm(range(len(slicing)-1), desc=f"{ei} frames"):
        t0, t1 = slicing[i], slicing[i+1]
        sv_array = sv2array(sv, time_idx_slice=slice(t0, t1), depth_idx_slice=slice(z_min_idx, z_max_idx), channels=channels)
        img = sv_array2image(sv_array, vmin, vmax, echogram_cmap)
        img.save(ei_save_path / f"{ei}_T{t0}-{t1}_Z{z_min_idx}-{z_max_idx}_Sv{vmin}-{vmax}.png")



# Dataset building tools

# Define config class 
@dataclass(frozen=True)
class DatasetConfig:
    time_frame_size: int
    vmin: float
    vmax: float
    z_min_idx: int
    z_max_idx : int
    frequencies: int | tuple[int, int, int] = (38, 70, 120)
    echogram_cmap: str = "RGB"
    correct_120: bool = False # whether to correct the depth treshold for the 120 kHz channel

    def name(self) -> str:
        freqs = [self.frequencies] if type(self.frequencies) == int else self.frequencies
        freqs = self.echogram_cmap +'_' + '_'.join(map(str, freqs)) +'kHz_'
        return (
            freqs +
            f'TF{self.time_frame_size}_' +
            f'Z{self.z_min_idx}-{self.z_max_idx}_' +
            f'Sv{self.vmin}-{self.vmax}dB'
        )

    def save_metadata(self, path:Path):
        with open(path / "dataset_config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)



# Dataset builder
def build_dataset(config: DatasetConfig,
                  ei_list: list[str],
                  root_path: Path):
    
    dataset_path = root_path / config.name()
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    config.save_metadata(dataset_path)

    for ei in ei_list:
        sv = load_survey_ds(survey=ei)["Sv"]
        plot_survey_RGB(sv=sv, 
                        frame_size=config.time_frame_size, 
                        z_min_idx=config.z_min_idx,
                        z_max_idx=config.z_max_idx, 
                        vmin=config.vmin, 
                        vmax=config.vmax, 
                        channels=config.frequencies,
                        echogram_cmap=config.echogram_cmap,
                        ei_save_path=dataset_path / ei, 
                        ei=ei)



if __name__ == '__main__':

    # Config grid builder
    FRAME_SIZES = [2_500, 5_000, 10_000]
    VMIN_VMAX = [(-90, -50), (-80, -50)]
    ZMIN_ZMAX = [(0, -1)]
    CHANNELS_CMAP = [((38, 70, 120), 'RGB'), (38, 'Greys_r')]

    def build_configs():
        for frame_size, (vmin, vmax), (zmin, zmax), (frequencies, echogram_cmap) in product(FRAME_SIZES, VMIN_VMAX, ZMIN_ZMAX, CHANNELS_CMAP):
            yield DatasetConfig(
                time_frame_size=frame_size,
                vmin=vmin,
                vmax=vmax,
                z_min_idx=zmin,
                z_max_idx=zmax,
                frequencies=frequencies,
                echogram_cmap=echogram_cmap
            )

    # Build all datasets in the configs built by build_configs()
    def build_all_datasets(ei_list, root_path):
        for config in build_configs():
            print(f"\nBuilding dataset: {config.name()}")
            build_dataset(
                config=config,
                ei_list=ei_list,
                root_path=root_path,
            )