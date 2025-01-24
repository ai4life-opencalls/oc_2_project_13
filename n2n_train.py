from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

from pathlib import Path
import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from cai_datasets import load_dataset

from careamics import CAREamist
from careamics.config import create_n2n_configuration

def split_even_odd_frames(x):
    """
        Splits odd and even frames of an array in even and odd frames.
        If the count of input frames is not even, the last frame is discarded.

        Args:

            - x (np.ndarray): An array of shape [N, C, H, W] or [C, H, W]
        Returns:
            - Tuple:
                - even_frames: np.ndarray of shape (N*C // 2, H, W)
                - odd_frames: np.ndarray of shape (N*C // 2, H, W)
        
    """
    # Use even frames as input and odd frames as target
    if x.ndim == 4:
        # concatenate stacks along the frame dimension
        x = np.concatenate(tuple(x))

    C, H, W = x.shape
    
    even_frames = np.stack([x[f] for f in list(range(0, C, 2))])
    odd_frames = np.stack([x[f] for f in list(range(1, C, 2))])

    # If the frames were odd to begin with, discard the last frame
    if C % 2 == 1:
        even_frames = even_frames[:-1]

    return even_frames, odd_frames




def train_n2n(train_dataset_name, 
              validation_dataset_name,
              dataset_folder, 
              models_folder, 
              experiment_name, 
              use_augmentations, 
              patch_size_z=None, 
              patch_size=64, 
              batch_size=16, 
              num_epochs=10,
              axes="SYX"):
    
    train_dataset_folder = os.path.join(dataset_folder, train_dataset_name)
    validation_dataset_folder = os.path.join(dataset_folder, validation_dataset_name)
    model_folder = os.path.join(models_folder, experiment_name)

    os.makedirs(model_folder, exist_ok=True)


    train_dataset = load_dataset(train_dataset_folder, stack=True)
    validation_dataset = load_dataset(validation_dataset_folder, stack=True)

    train_source, train_target = split_even_odd_frames(train_dataset)
    val_source, val_target = split_even_odd_frames(validation_dataset)

    config = create_n2n_configuration(
        experiment_name=experiment_name,
        data_type="array",
        axes=axes,
        patch_size=(patch_size_z, patch_size, patch_size) if patch_size_z is not None else (patch_size, patch_size),
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=[] if not use_augmentations else None
    )

    # instantiate a CAREamist
    careamist = CAREamist(
        source=config,
        work_dir=model_folder, 
    )

    # train
    careamist.train(
        train_source=train_source,
        train_target=train_target,
        val_source=val_source,
        val_target=val_target
    )


if __name__ == "__main__":
    
    # Get a parser that include some default ENV VARS overrides
    parser = get_argparser(description="Train a N2V model on the given dataset.")
    # Add script-specific varibles
    parser.add_argument('--train_dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--validation_dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--use_augmentations', action="store_true", help='Whether to use N2V2.')
    parser.add_argument('--patch_size_z', type=int, default=None, help="Patch depth dimension")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch spatial dimension")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Epochs to train")
    parser.add_argument('--axes', type=str, default="SYX", help="Axes used to interpret the TIFF files.")


    
    args = parser.parse_args()
    # Set Log Level from arguments
    log.setLevel(args.level)
    # Load env vars and args overrides into ENV dictionary
    load_env(args.env, parser_args=args)
    


    train_n2n(train_dataset_name = args.train_dataset_name,
              validation_dataset_name = args.validation_dataset_name,
              dataset_folder=ENV.get("DATASET_FOLDER"),
              models_folder=ENV.get("MODELS_FOLDER"),
              experiment_name=args.experiment_name,
              use_augmentations=args.use_augmentations,
              patch_size_z=args.patch_size_z, 
              patch_size=args.patch_size, 
              batch_size=args.batch_size, 
              num_epochs=args.num_epochs,
              axes=args.axes
              )
    
