from torch.utils.data import Dataset
from typing import Tuple
from predtiler.tile_stitcher import stitch_predictions
from predtiler.dataset import get_tiling_dataset, get_tile_manager
import tifffile
from pathlib import Path
import numpy as np


class CalciumImagingDataset(Dataset):
    def __init__(self, data_path, patch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.data = self.load_dataset(data_path)
        N, T, H, W = self.data.data.shape
        self.data = self.data.reshape(-1, H, W)

        self.dataset_mean = self.data.mean()
        self.dataset_std = self.data.std()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        n_idx, h, w = self.patch_location(index)
        # return the patch at the location (patch_size, patch_size)
        # do soiome normalization
        # 
        return (self.data[n_idx, h:h+self.patch_size, w:w+self.patch_size] - self.dataset_mean) / self.dataset_std, 0.0
    
    def patch_location(self, index:int)-> Tuple[int, int, int]:
        # it just ignores the index and returns a random location
        n_idx = np.random.randint(0,len(self.data))
        # c_idx = np.random.randint(0,self.data.shape[1])
        h = np.random.randint(0, self.data.shape[1]-self.patch_size)
        w = np.random.randint(0, self.data.shape[2]-self.patch_size)
        return (n_idx, h, w)

    def load_dataset(self, data_path):
        data = list()
        for p in list(Path(data_path).rglob("*.tif")):
            data.append(tifffile.imread(p))
        return np.array(data)


def make_predtiler_dataset(data_shape, tile_size, patch_size):    
    manager = get_tile_manager(data_shape=data_shape,
                            tile_shape=(1, tile_size, tile_size),
                            patch_shape=(1, patch_size, patch_size))
    return get_tiling_dataset(CalciumImagingDataset, manager)