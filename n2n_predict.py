from pathlib import Path
import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from careamics import CAREamist
from envutils import ENV, load_env, get_argparser, log

def predict_n2n(
                dataset_folder,
                models_folder,
                output_folder,
                experiment_name,
                remove_noise_from_pred=False,
                patch_size_z=None,
                patch_size=64, 
                batch_size=1,
                model_ckpt='last.ckpt',
                average_noise_tiff_name="average_image.tif"
                ):

    model_folder = models_folder.joinpath(experiment_name)
    ckpt_path = model_folder.joinpath('checkpoints', model_ckpt)

    print(f"Model folder is {model_folder}")
    exp_output_folder = output_folder.joinpath(experiment_name)
    print(f"Output Folder is {exp_output_folder}")
    average_noise_tiff_fp = dataset_folder.parent.joinpath(average_noise_tiff_name)
    print(f"Average Noise is {average_noise_tiff_fp}")
    
    train_dataset_path = dataset_folder.joinpath("train")
    val_dataset_path = dataset_folder.joinpath("val")
    test_dataset_path = dataset_folder.joinpath("test")

    average_noise = tifffile.imread(average_noise_tiff_fp) if remove_noise_from_pred else None

    careamist = CAREamist(ckpt_path, work_dir=model_folder)
    
    for in_data_path in [train_dataset_path, val_dataset_path, test_dataset_path]:
        # Appends ['train', 'val', 'test'] to the output path
        if not Path(in_data_path).exists():
            log.warning(f"Path {in_data_path} does not exist. Skipping...")
            continue
        out_data_path = exp_output_folder.joinpath(in_data_path.name)

        for tiff_fp in list(Path(in_data_path).rglob("*.tif")):
            img = tifffile.imread(tiff_fp)
            tile_size = (patch_size_z, patch_size, patch_size) if patch_size_z is not None else (patch_size, patch_size)
            prediction = careamist.predict(source=img,
                                           data_type="array",
                                            tile_size=tile_size,
                                            tile_overlap=[int(3/4*ts) for ts in tile_size],
                                            batch_size=batch_size,
                                            axes="SYX")
            prediction = np.array(prediction).squeeze()
            # Save the raw prediciton
            out_tiff_fp = out_data_path.joinpath(tiff_fp.name)
            out_tiff_fp.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(out_tiff_fp, prediction)

            # Save prediction without average noise
            if average_noise is not None:
                out_tiff_fp = out_data_path.parent.with_name(tiff_fp.parent.name + '_noval').joinpath(tiff_fp.name)
                out_tiff_fp.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(out_tiff_fp, prediction-average_noise)


if __name__ == "__main__":

    parser = get_argparser(description="Predict N2V for Calcium Imaging Denoising")

    parser.add_argument('-c', '--ckpt', '--checkpoint', type=str, default='last.ckpt', help="Model .ckpt filename to use for prediction. Defaults to last.ckpt.")
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--remove_noise_from_pred', action="store_true", help='Whether to save a version of the prediction with average noise subtracted. Average noise should be stored in the parent of the dataset folder, named as the average_noise_tiff_name parameter')
    parser.add_argument('--average_noise_tiff_name', type=str, default="average_image.tif", help='Name of the tiff file including average microscope noise.')
    parser.add_argument('--patch_size_z', type=int, default=None, help="Patch depth dimension")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch spatial dimension")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch Size")


    args = parser.parse_args()
    log.setLevel(args.level)
    load_env(args.env, parser_args=args)

    pred_gen = predict_n2n(
                dataset_folder=Path(ENV.get("DATASET_FOLDER")),
                models_folder=Path(ENV.get("MODELS_FOLDER")),
                output_folder=Path(ENV.get("OUTPUT_FOLDER")),
                experiment_name=args.experiment_name,
                remove_noise_from_pred=args.remove_noise_from_pred,
                patch_size_z=args.patch_size_z,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                model_ckpt=args.ckpt,
                average_noise_tiff_name=args.average_noise_tiff_name,
                )