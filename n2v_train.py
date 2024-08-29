from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import os
import glob
import tifffile
import matplotlib.pyplot as plt
import numpy as np

from careamics import CAREamist
from careamics.config import create_n2v_configuration

from careamics.lightning import TrainDataModule

DATASET_NAME = 'DeepCAD'
DATASET_FOLDER = os.getenv('DATASET_FOLDER')
MODELS_FOLDER = os.getenv('MODELS_FOLDER')
DATASET_SUBFOLDER = os.path.join(DATASET_FOLDER, DATASET_NAME)
#ALL_TIF = sorted(Path(DATASET_SUBFOLDER).glob('*.tif'))
use_n2v2 = False
algo = "n2v2" if use_n2v2 else "n2v"

experiment_name = f"{algo}_{DATASET_NAME}"


model_folder = os.path.join(MODELS_FOLDER, experiment_name)

os.makedirs(model_folder, exist_ok=True)


config = create_n2v_configuration(
    experiment_name=experiment_name,
    data_type="tiff",
    axes="ZYX",
    patch_size=(16, 64, 64),
    batch_size=1,
    num_epochs=10,
    use_n2v2=use_n2v2,
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

# prediction = careamist.predict(
#     source=train_image,
#     tile_size=(256, 256),
#     tile_overlap=(48, 48),
#     batch_size=1,
# )
# prediction = np.concatenate(prediction).squeeze()
# print(prediction.shape)

# tifffile.imwrite('try1result.tif', prediction)