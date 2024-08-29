import argparse
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration

import careamics.dataset.tiling as tiling
from careamics.prediction_utils import stitch_prediction_single
from typing import List
import logging

# Global vars
ENV = {"DATASET_FOLDER": None, "OUTPUT_FOLDER": None, "MODELS_FOLDER": None}
log = logging.getLogger(__name__)



def tiled_prediction(image: np.ndarray, model, patch_shape: List[int]=(16, 64, 64), patch_batch_size:int = 8):
    """
        N2V batched patch-wise prediction.
    """

    tiles_generator = tiling.extract_tiles(arr=image[None, None, ...], tile_size=patch_shape, overlaps=list(p//2 + 1 for p in patch_shape))
    
    pred_patches = list()
    patch_info = list()
    current_batch = list()
    while True:
        try:
            for b in range(patch_batch_size):
                current_tile, current_info = next(tiles_generator)
                patch_info.append(current_info)
                current_batch.append(current_tile)
        except StopIteration:
            break
        log.debug(f"Predicting batch {len(pred_patches)}...")
        pred_patches += model.predict(np.concatenate(current_batch), data_type='array', axes='SZYX')
        current_batch = list()

    pred_patches = list(np.array(pred_patches)[:, None, ...])
    return stitch_prediction_single(pred_patches, patch_info)


def main(dataset_name='DeepCAD', model_ckpt='last.ckpt', use_n2v2=False):
    
    # Variables and Paths
    dataset_subfolder = os.path.join(ENV['DATASET_FOLDER'], dataset_name)
    algo = "n2v2" if use_n2v2 else "n2v"

    experiment_name = f"{algo}_{dataset_name}"
    model_folder = os.path.join(ENV['MODELS_FOLDER'], experiment_name)
    ckpt_path = os.path.join(model_folder, 'checkpoints', model_ckpt)
    output_path = os.path.join(os.getenv('OUTPUT_FOLDER'), experiment_name)


    # instantiate a CAREamist
    careamist = CAREamist(
        ckpt_path,
        work_dir=model_folder, 
    )

    # Predicts over files
    for tiff_path_in in list(Path(dataset_subfolder).glob("*.tif*")):
        print(f"Predicting file {tiff_path_in}")
        tiff_out = tiled_prediction(tifffile.imread(tiff_path_in), careamist)
        tiff_path_out = Path.joinpath(Path(output_path), tiff_path_in.name)
        print(f"Writing prediction to {tiff_path_in}")
        os.makedirs(output_path, exist_ok=True)
        tifffile.imwrite(tiff_path_out, tiff_out)


### ENV MANAGEMENT AND ARGPARSE

def load_env(dot_env, env_dict=ENV):
    """
       Loads a .env file and validates the required environment variables, 
       allowing to keep track of environment variables used from your script and to manage multiple environments (e.g., multiple machines.)
       Envvars are loaded both in os.environ and in the ENV dictionary, you should use ENV dict since it is already validated (every envvar is not None).
       ENV dictionary must be defined as global variable.
    """
    if not load_dotenv(dot_env):
        raise FileNotFoundError(f"{dot_env} file not found. Please create a .env file or provide one using -e <env_filepath>. Has to contain the following envvars: {list(env_dict.keys())}")
    for k in env_dict:
        env_dict[k] = os.getenv(k)
        if env_dict[k] is None:
            raise ValueError(f"Environment Variable {k} has not been set in {dot_env}.")
        else:
            log.info(f"{k}: {env_dict[k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict N2V for Calcium Imaging Denoising")
    parser.add_argument('dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--use_n2v2', action='store_true', help="Loads the n2v2 model instead of the n2v one.")
    parser.add_argument('-c', '--ckpt', '--checkpoint', type=str, default='last.ckpt', help="Model .ckpt filename to use for prediction. Defaults to last.ckpt.")
    parser.add_argument('-e', '--env', type=str, default='.env', help="Path to an .env file containing required environment variables for the script")
    
    args = parser.parse_args()

    load_env(args.env)

    main(dataset_name=args.dataset_name,
         model_ckpt=args.ckpt,
         use_n2v2=args.use_n2v2)

    
    