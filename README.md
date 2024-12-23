# CalciumImagingDenoising
Application of different denoising models to a Calcium Imaging dataset.


## Setting Environment

Create a file '.env' in the current directory with the following environment variables:

```bash
    DATASET_FOLDER='your_path_to/calcium_imaging/' # Contains dataset Readme.m
    OUTPUT_FOLDER='output/'
    MODELS_FOLDER='models/'
```

you can also specify multiple `.custom_envs` and pass them to the script using `-e .custom_env`.

# Experiment Graph

Here follows the experiment map with the general overview of the experiment design.

![Experiment Graph](./docs/experiments_graph.drawio.svg)
