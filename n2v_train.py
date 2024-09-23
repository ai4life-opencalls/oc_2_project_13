from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

from pathlib import Path
import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration

from careamics.lightning import TrainDataModule


def train_n2v(dataset_name, 
              dataset_folder, 
              models_folder, 
              experiment_name, 
              use_n2v2, 
              use_augmentations, 
              patch_size_z=None, 
              patch_size=64, 
              batch_size=16, 
              num_epochs=10,
              axes="ZYX"):
    DATASET_SUBFOLDER = os.path.join(dataset_folder, dataset_name)

    model_folder = os.path.join(models_folder, experiment_name)

    os.makedirs(model_folder, exist_ok=True)

    config = create_n2v_configuration(
        experiment_name=experiment_name,
        data_type="tiff",
        axes=axes,
        patch_size=(patch_size_z, patch_size, patch_size) if patch_size_z is not None else (patch_size, patch_size),
        batch_size=batch_size,
        num_epochs=num_epochs,
        use_n2v2=use_n2v2,
        use_augmentations=use_augmentations
    )

    data_module = TrainDataModule( 
        data_config=config.data_config,
        train_data=DATASET_SUBFOLDER,
        use_in_memory=False
    )

    # instantiate a CAREamist
    careamist = CAREamist(
        source=config,
        work_dir=model_folder, 
    )

    # train
    careamist.train(
        datamodule=data_module,
        val_percentage=0.,
        val_minimum_split=100, # use 100 patches as validation
    )


if __name__ == "__main__":
    
    # Get a parser that include some default ENV VARS overrides
    parser = get_argparser(description="Train a N2V model on the given dataset.")
    # Add script-specific varibles
    parser.add_argument('--dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--use_n2v2', action="store_true", help='Whether to use N2V2.')
    parser.add_argument('--use_augmentations', action="store_true", help='Whether to use N2V2.')
    parser.add_argument('--patch_size_z', type=int, default=None, help="Patch depth dimension")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch spatial dimension")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Epochs to train")
    parser.add_argument('--axes', type=str, default="ZXY", help="Axes used to interpret the TIFF files.")


    
    args = parser.parse_args()
    # Set Log Level from arguments
    log.setLevel(args.level)
    # Load env vars and args overrides into ENV dictionary
    load_env(args.env, parser_args=args)
    


    train_n2v(dataset_name = args.dataset_name, 
              dataset_folder=ENV.get("DATASET_FOLDER"),
              models_folder=ENV.get("MODELS_FOLDER"),
              experiment_name=args.experiment_name,
              use_n2v2=args.use_n2v2,
              use_augmentations=args.use_augmentations,
              patch_size_z=args.patch_size_z, 
              patch_size=args.patch_size, 
              batch_size=args.batch_size, 
              num_epochs=args.num_epochs,
              axes=args.axes
              )
    
