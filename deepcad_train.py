from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

import torch
from pathlib import Path

from deepcad.train_collection import training_class
from deepcad.movie_display import display, display_img
from deepcad.utils import get_first_filename, download_demo


def train_deepcad(experiment_name, dataset_name, dataset_folder, models_folder, num_epochs, train_dataset_size, select_img_num, patch_size):
    
    n_epochs = num_epochs               # number of training epochs
    GPU = '0'                   # the index of GPU you will use (e.g. '0', '0,1', '0,1,2')
    train_datasets_size = train_dataset_size  # datasets size for training (how many 3D patches)
    patch_xyt = patch_size             # the width, height, and length of 3D patches (use isotropic patch size by default)
    overlap_factor = 0.4        # the overlap factor between two adjacent patches
    num_workers = 0             # if you use Windows system, set this to 0.

    # Setup some parameters for result visualization during training period (optional)
    save_test_images_per_epoch = True  # whether to save result images after each epoch

    dataset_path = Path(dataset_folder).joinpath(dataset_name)
    models_path = Path(models_folder).joinpath(experiment_name)
    models_path.mkdir(parents=True, exist_ok=True)

    train_dict = {
        # dataset dependent parameters
        'patch_x': patch_xyt,                          # the width of 3D patches
        'patch_y': patch_xyt,                          # the height of 3D patches
        'patch_t': patch_xyt,                          # the time dimension (frames) of 3D patches
        'overlap_factor':overlap_factor,               # the factor for image intensity scaling
        'scale_factor': 1,                             # the factor for image intensity scaling
        'select_img_num': select_img_num,                        # select the number of images used for training (use 2000 frames in colab)
        'train_datasets_size': train_datasets_size,    # datasets size for training (how many 3D patches)
        'datasets_path': str(dataset_path),                # folder containing files for training
        'pth_dir': str(models_path),                            # the path for pth file and result images
        
        # network related parameters
        'n_epochs': n_epochs,                          # the number of training epochs
        'lr': 0.00005,                                 # learning rate
        'b1': 0.5,                                     # Adam: bata1
        'b2': 0.999,                                   # Adam: bata2
        'fmap': 16,                                    # model complexity
        'GPU': GPU,                                    # GPU index
        'num_workers': num_workers,                    # if you use Windows system, set this to 0.
        'visualize_images_per_epoch': False,                       # whether to show result images after each epoch
        'save_test_images_per_epoch': save_test_images_per_epoch,  # whether to save result images after each epoch
        'colab_display': False
    }
    tc = training_class(train_dict)

    tc.run()

if __name__ == "__main__":

     # Get a parser that include some default ENV VARS overrides
    parser = get_argparser(description="Train a DeepCAD model on the given Dataset")
    parser.add_argument('--dataset_name', type=str, help='Dataset Name, as subfolder of the dataset directory containing the .tif files')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--train_dataset_size', type=int, default=10000, help='datasets size for training (how many 3D patches)')
    parser.add_argument('--select_img_num', type=int, default=20000, help="Frames to select")
    parser.add_argument('--patch_size', type=int, default=150, help='Isotropic dimension of 3D patches')
    parser.add_argument('--num_epochs', type=int, default=5, help="Epochs to train")

    


    args = parser.parse_args()
    # Set Log Level from arguments
    log.setLevel(args.level)
    # Load env vars and args overrides into ENV dictionary
    load_env(args.env, parser_args=args)

    train_deepcad(
                    experiment_name=args.experiment_name,
                    dataset_name=args.dataset_name,
                    dataset_folder=ENV.get("DATASET_FOLDER"),
                    models_folder=ENV.get("MODELS_FOLDER"),
                    train_dataset_size=args.train_dataset_size,
                    select_img_num=args.select_img_num,
                    patch_size=args.patch_size,
                    num_epochs=args.num_epochs,
                 )