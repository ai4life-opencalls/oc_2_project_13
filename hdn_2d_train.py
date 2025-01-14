import sys
sys.path.append('hdn')

from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

from pathlib import Path
import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy


from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
from torch.utils.data import DataLoader

from datasets import make_predtiler_dataset

from hdn.models.lvae import LadderVAE
from hdn import training as hdn_training


def train_2d_hdn(train_dataset_name, 
              validation_dataset_name,
              test_dataset_name,
              dataset_folder, 
              models_folder, 
              experiment_name,
              patch_size,
              tile_size,
              batch_size,
              num_epochs,
              data_channels: int,
              data_hw: int,
              lr=3e-4,
              device = "cuda"
              ):
    """
        Train an Hierchical DivNoising Model    

        Args:
            
            - data_channels: Length of the time / channel dimension. Needed for tiling
            - data_hw: int. Spatial dimension, assuming square frames.

    """
    train_dataset_folder = os.path.join(dataset_folder, train_dataset_name)
    validation_dataset_folder = os.path.join(dataset_folder, validation_dataset_name)
    test_dataset_folder = os.path.join(dataset_folder, test_dataset_name)

    model_folder = os.path.join(models_folder, experiment_name)

    os.makedirs(model_folder, exist_ok=True)

    data_shape = [data_channels, data_hw, data_hw]
    
    gmm_to_load = Path(model_folder).joinpath("noise_model", "GMM.npz")
    hdn_model_folder = Path(model_folder).joinpath("hdn")

    noise_model = GaussianMixtureNoiseModel(    
                                    path=Path(gmm_to_load).parent, 
                                    device = device, 
                                    params = np.load(gmm_to_load, allow_pickle=True)
                                )
    
    PatchedCalciumImagingDataset = make_predtiler_dataset(data_shape=data_shape,
                                                        tile_size=tile_size,
                                                        patch_size=patch_size)

    
    train_dataset = PatchedCalciumImagingDataset(train_dataset_folder, patch_size=patch_size)
    val_dataset  = PatchedCalciumImagingDataset(validation_dataset_folder, patch_size=patch_size)
    test_dataset  = PatchedCalciumImagingDataset(test_dataset_folder, patch_size=patch_size)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    num_latents = 6
    z_dims = [32]*int(num_latents)
    blocks_per_layer = 5
    batchnorm = True
    free_bits = 1.0
    use_uncond_mode_at=[0,1]

    hdn_model = LadderVAE(z_dims=z_dims,
                    blocks_per_layer=blocks_per_layer,
                    data_mean=train_dataset.dataset_mean,
                    data_std=train_dataset.dataset_std,
                    noiseModel=noise_model,
                    device=device,
                    batchnorm=batchnorm,
                    free_bits=free_bits,
                    img_shape=[patch_size, patch_size],
                    use_uncond_mode_at=use_uncond_mode_at).to(device=device)

    hdn_model.train() # Model set in training mode

    hdn_training.train_network(model=hdn_model,
                        lr=lr,
                        max_epochs=num_epochs,
                        steps_per_epoch=len(train_loader),
                        directory_path=str(hdn_model_folder),
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        virtual_batch=batch_size,
                        gaussian_noise_std=None,
                        model_name=experiment_name,
                        val_loss_patience=30)

if __name__ == "__main__":
    
    # Get a parser that include some default ENV VARS overrides
    parser = get_argparser(description="Train a N2V model on the given dataset.")
    # Add script-specific varibles
    parser.add_argument('--train_dataset_name', default="train", type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--validation_dataset_name', default="val", type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--test_dataset_name', default="test", type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--patch_size', type=int, default=64, help="Patch spatial dimension")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser.add_argument('--tile_size', type=int, default=32, help="Tile Size for tiled prediction")
    parser.add_argument('--num_epochs', type=int, default=500, help="Epochs to train")
    parser.add_argument('--data_channels', type=int, default=1009, help="Length of channel or time dimension")
    parser.add_argument('--data_hw', type=int, default=1024, help="Spatial dimension, assuming square frames.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning Rate")
    
    args = parser.parse_args()
    # Set Log Level from arguments
    log.setLevel(args.level)
    # Load env vars and args overrides into ENV dictionary
    load_env(args.env, parser_args=args)

    train_2d_hdn(train_dataset_name = args.train_dataset_name,
                    validation_dataset_name = args.validation_dataset_name,
                    test_dataset_name=args.test_dataset_name,
                    dataset_folder=ENV.get("DATASET_FOLDER"),
                    models_folder=ENV.get("MODELS_FOLDER"),
                    experiment_name=args.experiment_name,
                    patch_size=args.patch_size, 
                    tile_size=args.tile_size,
                    batch_size=args.batch_size, 
                    num_epochs=args.num_epochs,
                    data_channels=args.data_channels,
                    data_hw=args.data_hw,
                    lr=args.lr
              )
    
