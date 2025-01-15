from envutils import ENV, load_env, get_tiff_paths, get_argparser, log

from hdn.lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from hdn.lib import histNoiseModel
from hdn.lib.utils import plotProbabilityDistribution

import tifffile
import numpy as np
from pathlib import Path



def train_noise_models(signal_folder: str,
                       denoised_folder: str,
                       models_folder: str,
                       experiment_name: str,
                       n_coeff: int = 2,
                       n_gaussian: int = 3,
                       gmm_epochs: int = 1000,
                       histogram_bins: int = 256,
                       random_perc: float = 1.0,
                       gmm_learning_rate: float = 0.1,
                       gmm_batch_size: int = 250000,
                       gmm_clip_perc: float = 0.1,
                       gmm_min_sigma: float = 50,
                       normalize_data: bool = False,
                       device: str = 'cuda'):
    
    """
        Train an Histogram Noise Model and a Gaussian Mixture Model on the given signal and denoised images.
        
        Args:
        
            signal_folder: str
                Folder containing the signal images
            denoised_folder: str
                Folder containing the denoised images
            models_folder: str
                Folder where the noise models will be saved
            experiment_name: str
                Name of the experiment. Will be used to create corresponding subfolders.
            n_coeff: int
                Number of coefficients for the GMM
            n_gaussian: int
                Number of Gaussians for the GMM
            gmm_epochs: int
                Number of epochs for the GMM
            histogram_bins: int
                Number of bins for the histogram
            random_perc: float
                Percentage of the dataset to use for training
            gmm_learning_rate: float
                Learning rate for the GMM
            gmm_batch_size: int
                Batch size for the GMM
            gmm_clip_perc: float
                Percentage to clip for the GMM
            gmm_min_sigma: float
                Minimum sigma for the GMM
            device: str
                Device to use for training    
            normalize_data: str
                Whether to normalize data before training.
    """

    noise_model_folder = Path(models_folder).joinpath(experiment_name, "noise_model")
    noise_model_folder.mkdir(parents=True, exist_ok=True)
    print(f"Noise models will be saved to {noise_model_folder}")

    # Load signal and denoised images
    # Original dataset
    signal_tiff = list(Path(signal_folder).rglob("*.tif"))
    denoised_tiff = list(Path(denoised_folder).rglob("*.tif"))

    print(f"Found signals {signal_tiff}")
    print(f"Found denoised {denoised_tiff}")

    input_tiff = list()
    # Ensure signal and denoised files are loaded together
    for stiff in signal_tiff:
        dtiff = [t for t in list(denoised_tiff) if t.name == stiff.name][0]
        input_tiff.append((stiff, dtiff))
        print(f"Matching {stiff} with {dtiff}")

    signal = []
    denoised = []

    for tsig, tden in zip(signal_tiff, denoised_tiff):
        print(f"Loading {stiff} and {dtiff}")
        signal.append(tifffile.imread(tsig).flatten())
        denoised.append(tifffile.imread(tden).flatten())
    
    print(f"Concatenating files...")
    signal = np.concatenate(signal, axis=0)
    denoised = np.concatenate(denoised, axis=0)
    # Here signal and denoised are 1D arrays with all the pixels concatenated

    if normalize_data:
        signal = (signal - signal.mean()) / signal.std() 
        denoised = (denoised - denoised.mean()) / denoised.std()

    minval, maxval = signal.min(), signal.max()

    print(f"Sampling Data...")
    if random_perc < 1.0:
        print(f"Using {random_perc * 100}% of the data")
        idx = np.random.choice(signal.shape[0], int(random_perc * signal.shape[0]), replace=False)
        signal = signal[idx]
        denoised = denoised[idx]

    # print(f"Training Histogram...")
    # # Train an Histogram Noise Model
    # histogram = histNoiseModel.createHistogram(bins=histogram_bins, 
    #                                        minVal=minval, 
    #                                        maxVal=maxval, 
    #                                        observation=denoised, 
    #                                        signal=signal)

    # # Create output folder and save histogram
    # hist_savepath = str(Path(noise_model_folder).joinpath('histogram.npy'))
    # np.save(hist_savepath, histogram)
    # print(f"Saved histogram to {hist_savepath}")

    gmm_savepath = str(Path(noise_model_folder)) + '/'
    print("Training GMM")
    gaussianMixtureNoiseModel = GaussianMixtureNoiseModel(min_signal = minval, 
                                                          max_signal = maxval, 
                                                          path=gmm_savepath, 
                                                          weight = None, 
                                                          n_gaussian = n_gaussian, 
                                                          n_coeff = n_coeff, 
                                                          device = device, 
                                                          min_sigma = gmm_min_sigma)

    # Train GMM
    gaussianMixtureNoiseModel.train(signal, 
                                    denoised, 
                                    batchSize = gmm_batch_size,
                                    n_epochs = gmm_epochs, 
                                    learning_rate = gmm_learning_rate, 
                                    name = 'GMM', 
                                    lowerClip = 100.0 - gmm_clip_perc, 
                                    upperClip = gmm_clip_perc,
                                    )




if __name__ == "__main__":
    
    # Get a parser that include some default ENV VARS overrides
    parser = get_argparser(description="Train a Noise Model on the given signal and observation datasets.")
    # Add script-specific varibles
    parser.add_argument('--signal_folder', type=str, help='Folder containing the signal images')
    parser.add_argument('--denoised_folder', type=str, help='Folder containing the denoised images')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment. Will be used to create corresponding subfolders.')
    parser.add_argument('--histogram_bins', type=int, default=256, help='Number of bins for the histogram')
    parser.add_argument('--n_coeff', type=int, default=2, help='Number of coefficients for the GMM')
    parser.add_argument('--n_gaussian', type=int, default=3, help='Number of Gaussians for the GMM')
    parser.add_argument('--gmm_epochs', type=int, default=1000, help='Number of epochs for the GMM')
    parser.add_argument('--gmm_learning_rate', type=float, default=0.1, help='Learning rate for the GMM')  
    parser.add_argument('--gmm_batch_size', type=int, default=250000, help='Batch size for the GMM')
    parser.add_argument('--gmm_clip_perc', type=float, default=0.1, help='Percentage to clip for the GMM')
    parser.add_argument('--gmm_min_sigma', type=float, default=50, help='Minimum sigma for the GMM')
    parser.add_argument('--random_perc', type=float, default=1.0, help='Percentage of the dataset to use for training')
    parser.add_argument('--normalize_data', action="store_true", help="Whether to normalize data before training")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')

    args = parser.parse_args()
    # Set Log Level from arguments
    log.setLevel(args.level)
    # Load env vars and args overrides into ENV dictionary
    load_env(args.env, parser_args=args)
    print("env loaded", flush=True)

    train_noise_models( 
                        signal_folder = args.signal_folder,
                        denoised_folder = args.denoised_folder,
                        models_folder=ENV.get("MODELS_FOLDER"),
                        experiment_name=args.experiment_name,
                        n_coeff=args.n_coeff,
                        n_gaussian=args.n_gaussian,
                        gmm_epochs=args.gmm_epochs,
                        histogram_bins=args.histogram_bins,
                        random_perc=args.random_perc,
                        device=args.device,
                        gmm_learning_rate=args.gmm_learning_rate,
                        gmm_batch_size=args.gmm_batch_size,
                        gmm_clip_perc=args.gmm_clip_perc,
                        gmm_min_sigma=args.gmm_min_sigma,
                        normalize_data=args.normalize_data
                     )

    