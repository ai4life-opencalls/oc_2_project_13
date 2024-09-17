from envutils import ENV, load_env, log, get_tiff_paths, add_default_arguments

import argparse


import matplotlib.pyplot as plt
import numpy as np
from careamics.utils import autocorrelation
import os
import tifffile
from pathlib import Path

EXPERIMENT_NAME = "ACORR_2"

def save_autocorrelations(dataset_name, size=50, n_frames=5, env:dict=ENV):

    OUT_FOLDER = Path(f'{env["OUTPUT_FOLDER"]}').joinpath(EXPERIMENT_NAME, dataset_name)
    os.makedirs(OUT_FOLDER, exist_ok=True)

    N = size

    for tiff_path in get_tiff_paths(dataset_name):
        image = tifffile.imread(tiff_path)
        ac = autocorrelation(image=image)
        
        fig, ax = plt.subplots(ncols=n_frames, figsize=(16, 4))
        
        # Calculate the center coordinates of the autocorrelation result
        center_y, center_x = ac.shape[1] // 2, ac.shape[2] // 2
        half_N = N // 2
        start_x, start_y = max(0, center_x - half_N), max(0, center_y - half_N)
        end_x = min(ac.shape[2], center_x + half_N)
        end_y = min(ac.shape[1], center_y + half_N)
        


        for i, f in enumerate(np.linspace(0, ac.shape[0]-1, 5).astype(np.int32)):
            # Crop the NxN region from the center of each autocorrelation frame
            cropped_ac = ac[f, start_y:end_y, start_x:end_x]
            ax[i].imshow(cropped_ac, cmap='gray')

            ax[i].axis('off')
            ax[i].set_title(f"Frame {f}")
        fig.suptitle(f"{tiff_path.name}")
        fig.tight_layout()
        fig.savefig(OUT_FOLDER.joinpath(tiff_path.name.replace(".tif", ".png")))
        plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=f"Plot Autocorrelations for the given video dataset. REQUIRED ENV VARS: {list(ENV.keys())}")
    add_default_arguments(parser)
    parser.add_argument('-s', '--ac_size', type=int, default=50, help='Size of the autocorrelation around the center.')
    parser.add_argument('-f', '--frames', type=int, default=5, help='Number of equally-spaced frames to show.')
    args = parser.parse_args()
    load_env(args.env)

    save_autocorrelations(dataset_name=args.dataset_name,
         size=args.ac_size,
         n_frames=args.frames,
         env=ENV
         )