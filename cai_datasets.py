from torch.utils.data import Dataset
from typing import Tuple
from predtiler.tile_stitcher import stitch_predictions
from predtiler.dataset import get_tiling_dataset, get_tile_manager
import tifffile
from pathlib import Path
import numpy as np


def load_dataset(data_path, stack=True):
    """
        Load all '*.tif' files contained in the given folder.

        Args: 
            - data_path (Path or str): Path of the folder containing the files
            - stack (bool): If true, returns a numpy array with the images stacked together. Otherwise returns a list.
        
        Returns:
            - (List or np.ndarray): Loaded images. If concat is false, a List of ndarrays. Otherwise, a single ndarray of shape (N, H, W) or (N, C, H, W).
    """
    data = list()
    for p in list(Path(data_path).rglob("*.tif")):
        data.append(tifffile.imread(p))
    if stack:
        return np.array(data)
    else:
        return data


class CalciumImagingDataset(Dataset):
    def __init__(self, data_path, patch_size=64, normalize=True):
        super().__init__()
        self.patch_size = patch_size
        self.data = load_dataset(data_path, stack=True)
        N, T, H, W = self.data.data.shape
        self.data = self.data.reshape(-1, H, W)
        self.normalize = normalize
        self.dataset_mean = self.data.mean()
        self.dataset_std = self.data.std()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        n_idx, h, w = self.patch_location(index)
        # return the patch at the location (patch_size, patch_size)
        # do soiome normalization
        # 
        if self.normalize:
            return self.data[n_idx, h:h+self.patch_size, w:w+self.patch_size] - self.dataset_mean, 0.0
        else:
            return self.data[n_idx, h:h+self.patch_size, w:w+self.patch_size], 0.0

    def patch_location(self, index:int)-> Tuple[int, int, int]:
        # it just ignores the index and returns a random location
        n_idx = np.random.randint(0,len(self.data))
        # c_idx = np.random.randint(0,self.data.shape[1])
        h = np.random.randint(0, self.data.shape[1]-self.patch_size)
        w = np.random.randint(0, self.data.shape[2]-self.patch_size)
        return (n_idx, h, w)

    


def make_predtiler_dataset(data_shape, tile_size, patch_size):    
    manager = get_tile_manager(data_shape=data_shape,
                            tile_shape=(1, tile_size, tile_size),
                            patch_shape=(1, patch_size, patch_size))
    return get_tiling_dataset(CalciumImagingDataset, manager)