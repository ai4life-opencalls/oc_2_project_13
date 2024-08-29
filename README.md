# CalciumImagingDenoising
Application of different denoising models to Calcium Images


## Setting Environment

Create a file '.env' in the current directory with the following environment variables:

```bash
    DATASET_FOLDER='your_path_to/calcium_imaging/' # Contains dataset Readme.md
    OUTPUT_FOLDER='output/'
    MODELS_FOLDER='models/'
```

you can also specify multiple `.custom_envs` and pass them to the script using `-e .custom_env`.

## N2V

### Training

Run `python n2v_train.py`.

### Prediction

Run `python n2v_predict.py`