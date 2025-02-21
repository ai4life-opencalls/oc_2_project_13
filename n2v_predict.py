import argparse
from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

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


def tiled_prediction(image: np.ndarray, model, patch_shape: List[int]=(16, 64, 64), patch_batch_size:int = 8, axes="ZYX"):
    """
        N2V batched patch-wise prediction.
    """

    model_2d = len(patch_shape) == 2

    if model_2d:
        # Predict frame by frame to avoid OOM
        pred_frames = list()
        for batch_start in range(0, image.shape[0], patch_batch_size):
            print(f"Predicting batch {batch_start} to {min(batch_start + patch_batch_size, image.shape[0])} of {image.shape[0]}")
            batch_end = min(batch_start + patch_batch_size, image.shape[0])
            current_batch = image[batch_start:batch_end]
            pred_batch = model.predict(current_batch, data_type='array', axes=axes, batch_size=current_batch.shape[0])
            pred_frames += pred_batch
        return np.concat(pred_frames, axis=1).squeeze()

    else:

        tiles_generator = tiling.extract_tiles(arr=image[None, None, ...], tile_size=patch_shape, overlaps=[int(3/4*ts) for ts in patch_shape])
        
        pred_patches = list()
        patch_info = list()
        current_batch = list()
        
        for current_tile, current_info in tiles_generator:
            patch_info.append(current_info)
            current_batch.append(current_tile)
            predict_axes = f'SZYX' # if 'Z' in axes or 'T' in axes else f'CYX'
            # Processing batch (if full)
            if len(current_batch) == patch_batch_size:
                pred_patches += model.predict(np.concatenate(current_batch), data_type='array', axes=predict_axes)
                current_batch = []  # Create new empty batch 
                log.debug(f"Predicted {len(pred_patches)} tiles...")
        # Process last batch if is n_tiles % batch_size > 0
        if len(current_batch) > 0:
            pred_patches += model.predict(np.concatenate(current_batch), data_type='array', axes=predict_axes)
            current_batch = []  # Create new empty batch
            log.debug(f"Predicted {len(pred_patches)} tile. Done.")
    pred_patches = list(np.array(pred_patches)[:, None, ...])
    return stitch_prediction_single(pred_patches, patch_info)


def predict_n2v(
         dataset_name, 
         dataset_folder,
         models_folder,
         output_folder,
         experiment_name,
         save_outputs=True,
         patch_size_z=None,
         patch_size=64, 
         batch_size=16,
         axes="YX",
         model_ckpt='last.ckpt',
         use_custom_tiled_prediction=False
         ):
    
    """
        Predict N2V on the given dataset.

        Args:
            - dataset_name: subfolder of dataset_folder containing the .tif[f] files
            - dataset_folder: dataset folder
            - models_folder: models folder. Must contain a folder named as experiment_name, containing the output of careamics training.
            - output_folder: Where to save outputs, if save_output is True (default). A subfolder named as experiment_name will be created.
            - experiment_name: Name of the experiment, used both to load models and to output results.
            - save_outputs: If True, the function just iterates over the dataset and saves predictions as tiff file in output_folder/<experiment_name>. Otherwise, it yields a tuple (input_file_path, input_tiff, predicted_tiff)
            - patch_size_z: The depth dimension of patchwise prediction. If model is 3d, it should be None.
            - patch_size: Size of patchwise prediction.
            - batch_size
            - axes: Axes argument that will be passed to the model. Must match with those used during training.
            - model_ckpt: checkpoint that will be used in the `checkpoints` folder. defaults to last.ckpt
            - use_custom_tiled_prediction: If True, uses a custom tiled prediction function. May help if the standard prediction goes OOM.

    """
    
    # Variables and Paths
    model_folder = os.path.join(models_folder, experiment_name)
    ckpt_path = os.path.join(model_folder, 'checkpoints', model_ckpt)
    output_path = os.path.join(output_folder, experiment_name, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # instantiate a CAREamist
    careamist = CAREamist(
        ckpt_path,
        work_dir=model_folder, 
    )

    # Predicts over files
    for tiff_path_in in sorted(Path(dataset_folder).joinpath(dataset_name).glob(f'*.tif*')):

        print(f"Predicting file {tiff_path_in}")
        tiff_in = tifffile.imread(tiff_path_in)
        tile_size = (patch_size_z, patch_size, patch_size) if patch_size_z is not None else (patch_size, patch_size)

        
        if use_custom_tiled_prediction:
            tiff_out = tiled_prediction(
                                        image=tiff_in,
                                        model=careamist,
                                        patch_shape=(patch_size_z, patch_size, patch_size) if patch_size_z is not None else (patch_size, patch_size),
                                        patch_batch_size=batch_size,
                                        axes=axes
                                        )
        else:
            tiff_out = careamist.predict(source=tiff_in,
                                    data_type="array",
                                    tile_size=tile_size,
                                    tile_overlap=[int(3/4*ts) for ts in tile_size],
                                    batch_size=batch_size,
                                    axes=axes)
            
        tiff_out = np.squeeze(tiff_out)

        if save_outputs:
            tiff_path_out = Path.joinpath(Path(output_path), tiff_path_in.name)
            print(f"Writing prediction to {tiff_path_out}")
            os.makedirs(output_path, exist_ok=True)
            tifffile.imwrite(tiff_path_out, tiff_out)
        else:
            yield (tiff_path_in, tiff_in, tiff_out)

if __name__ == "__main__":

    parser = get_argparser(description="Predict N2V for Calcium Imaging Denoising")

    parser.add_argument('-c', '--ckpt', '--checkpoint', type=str, default='last.ckpt', help="Model .ckpt filename to use for prediction. Defaults to last.ckpt.")
    parser.add_argument('--dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--patch_size_z', type=int, default=None, help="Patch depth dimension")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch spatial dimension")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch Size")
    parser.add_argument('--axes', type=str, default="YX", help="Axes used to interpret the TIFF files.")
    parser.add_argument('--use_custom_tiled_prediction', action='store_true', help="Use custom tiled prediction function instead of CAREamics built-in. May help if the standard prediction goes OOM.")

    args = parser.parse_args()
    log.setLevel(args.level)
    load_env(args.env, parser_args=args)

    pred_gen = predict_n2v(dataset_name=args.dataset_name,
                dataset_folder=ENV.get("DATASET_FOLDER"),
                models_folder=ENV.get("MODELS_FOLDER"),
                output_folder=ENV.get("OUTPUT_FOLDER"),
                experiment_name=args.experiment_name,
                patch_size_z=args.patch_size_z,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                axes=args.axes,
                model_ckpt=args.ckpt,
                use_custom_tiled_prediction=args.use_custom_tiled_prediction
                )
    for _ in pred_gen:
        log.info("Done.")

    
    