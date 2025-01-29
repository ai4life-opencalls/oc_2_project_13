<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true">
  </a>
</p>

# Project 13: Optimizing calcium image acquisition with Machine Learning denoising algorithms

# Overview

This repository contains code for running multiple denoising algorithms on a Calcium Imaging dataset. The algorithms used in this repository are Noise2Noise (N2N), Noise2Void (N2V), Hierarchical DivNoising (HDN). Partial results are also provided for DeepCAD and CellMincer.

The goal of this repository is to analyze the best strategy for denoising Calcium Imaging data in the context of limited training data, high noise levels and no access to clean reference images. 

For this reason, this project is to be intended as a non-exaustive exploration of imaging denoising algorithms in the context of Calcium Imaging data. \
In particular, the focus is on providing researchers with a set of ready-to-use tools. For this reason we prioritized algorithms that promoted accessibility, while also providing instructions and fixes for codebases that required more effort to be run.

For each concluded experiment, we provide a set of notebooks and the standalone scripts that can be readily used and adapted to the user's needs.

## Getting started

Due to the computational requirements of some of the algoritms used, part of the code hosted in this repository has been designed to run as standalone scripts (e.g., on an HPC).
Tutorials notebooks for the most relevant experiments are also provided. They are named as the experiment code reported in the following section.

### Setting Environment

Create a file '.env' in the current directory with the following environment variables:

```bash
    DATASET_FOLDER='your_path_to/calcium_imaging/' # Root folder of this repository
    OUTPUT_FOLDER='output/'
    MODELS_FOLDER='models/'
```

you can also specify multiple `.custom_envs` and pass them to the script using `-e .custom_env`.

# Dataset and pre-processing

The dataset used in this project has been provided by researchers at McGill University in the context of AI4Life Open Challenges 2024.

The dataset consisted of HUVEC cells with GCaMP6s. 20ms exposure, 20% lamp intensity, stream acquisition with widefield imaging. 

The original dataset consisted of three folders, one named "DeepCAD", containing 30 1009x1024x1024 videos, one named "Test Data", containing 30 1009x1024x1024 videos, and the one named "NIDDL", containing 163 Low-SNR and 163 High-SNR paired frames. Both DeepCAD and NIDDL are made from a combination of 3 different experiments made with the same imaging conditions. \
We used the "NIDDL" for performing some hyperparameter tuning and testing the codebase, while data from "DeepCAD" and "Test Data" was used for main experiments. \
Additionally, the applicants reported to have previously subtracted the average microscope background and provided a single frame of noise. Before proceeding with the experiments, we added back the background noise to retrieve RAW data and ensure consistency between the experiments.

As detailed in the experiment map in the following section, we discovered that most of the videos we received were affected by an acquisition or quantization issue that caused the dynamic range of the images to be overly compressed (e.g., only 4-5 pixel intensities were used to express the signal). For this reason, we proceeded to manually select videos that were less affected by this issue, resulting in the following **REFINED** dataset, (composed of only images originally from the "TestData" folder):

```
Training:
    - 20ms_20%_Yoda1_005.tif
    - 20ms_20%_Yoda1_006.tif
Validation:
    - 20ms_20%_Yoda1_008.tif
Testing:
    - 20ms_20%_Yoda1_009.tif
```

# Experiment Overview

Here follows the experiment roadmap with the general overview of the experiment design. \
This is intended as a guide to navigate the repository and understand the experiments that have been run and the possible future directions. \
For the experiments that has been run in a standalone script, the run command is provided in the respective notebook. \
For Noise2Noise and Noise2Void we use the ![CAREamics](https://careamics.github.io/) implementation. 

![Experiment Graph](./docs/experiments_graph.drawio.svg)

# Algorithm Details

## Noise2Noise (N2N)

![Paper](https://arxiv.org/abs/1803.04189)

### Overview

Noise2Void (N2V) is a self-supervised denoising method that learns to restore images without requiring clean reference data. It operates by randomly masking pixels in an image and predicting their values based on surrounding pixel information. N2V is built on the assumption that underlying image structures are smooth, while noise is pixel-wise independent. This allows it to estimate true signal values from neighboring pixels while ensuring noise remains unpredictable, leading to effective denoising in cases where these assumptions hold.

### Details

N2V utilizes a UNet-based architecture and applies a specialized data augmentation technique. This involves replacing randomly selected pixels with values from neighboring pixels, creating a training scenario where the model learns to reconstruct the original pixel values. The network is trained by computing a loss only on these altered pixels, reinforcing its ability to infer the true signal while disregarding noise. This self-supervised approach eliminates the need for paired noisy-clean datasets, making it highly adaptable to diverse real-world scenarios.

## Noise2Void (N2V)

![Paper](https://arxiv.org/abs/1811.10980)

### Overview

Noise2Void operates under a self-supervised learning paradigm, exploiting the statistical properties of noise and signal in images. The core assumption is that while the signal (the denoising target) is not pixel-wise independent, the noise is conditionally independent given the signal. This allows N2V to extract meaningful information from a single noisy image by focusing on the relationship between neighboring pixels rather than relying on an exact match with a clean target.

### Details
The architecture employed in N2V is based on blind-spot networks. This design excludes the central pixel from its receptive field during training. 
During training, a mask prevents the model from accessing the value of the central pixel, that the network aims to predict. Instead, it utilizes information from surrounding pixels, which encourages the model to learn how to reconstruct the central pixel's value based on its context within the image. In this repository, we used both 2D and 3D U-Net architectures for denoising Calcium Imaging data.

### Advantages
One of the most significant advantages of Noise2Void is that it doesn't require ground truth images or high SNR images to be trained. This characteristic makes N2V particularly suitable for denoising microscopy images, where obtaining clean reference images can be challenging. Moreover, N2V has been shown to perform competitively against other denoising methods that do have access to clean targets or noisy pairs, despite its more limited training data requirements.

## Hierarchical DivNoising (HDN)

### Overview
Hierarchical DivNoising (HDN) is another advanced approach in image denoising that builds on concepts similar to N2V but introduces a hierarchical structure for improved performance. HDN is also capable to remove spatially-correlated noise, however this procedure requires a pre-processing step in which noise is estimated from a noisy observation and a high SNR signal.

### Details
HDN organizes the denoising process into multiple levels, allowing it to capture features at varying scales. This hierarchical approach enhances the model's ability to understand complex patterns within noisy images, improving denoising efficacy. Moreover, 

### Advantages
The hierarchical structure of HDN facilitates better contextual understanding and feature extraction compared to traditional methods. This can lead to better results in scenarios with significant noise levels or complex image structures, making HDN a valuable tool in applications requiring high-quality image restoration.


### Other Algorithms
Here follows a brief overview of the other algorithms that is worth investigating in the context of Calcium Imaging denoising. 


## DeepCAD

![Paper](https://www.nature.com/articles/s41592-021-01225-0)

### Overview

DeepCAD is a deep learning-based denoising algorithm designed specifically for Calcium Imaging data. DeepCAD is based on the assumption that in the context of Calcium Imaging sampled at 30 Hz, two consecutive frames are highly correlated and can be regarded as two independent samples of the same underlying signal. This allows the model to leverage the temporal information present in the data by using frame pairs as input and target pairs during training. 

### Details

In DeepCAD, the model architecture consists of a 3D U-Net, trained with a self-supervised strategy in which the input and output are pairs of consecutive frames. In this way, the model learns to predict the next frame in the sequence based on the current frame, effectively denoising the data by exploiting the temporal correlations between frames.

## CellMincer

![Paper](https://cellmincer.readthedocs.io/en/latest/installation/index.html)

### Overview

CellMincer is a self-supervised framework designed to work with Voltage Imaging. designed specifically for denoising voltage imaging datasets.
CellMincer operates by masking and predicting sparse sets of pixels over short temporal windows. This allows the model to leverage both the unique information from individual frames and the context from neighboring frames. 

### Details
The architecture consists of a frame-wise 2D U-Net module for spatial feature extraction followed by a pixelwise 1D convolutional module for temporal data processing, optimizing both spatial and temporal correlations of the data.

The authors report that the performance of CellMincer on Calcium Imaging data are degraded compared to those of DeepCAD. This may be due to the different scales in temporal dynamics between Voltage and Calcium imaging. Nonetheless, this frameword remains a direction worth investigating.


## Results Overview

For a detailed comparison of the results obtained with each algorithm, please refer to the respective experiment notebooks. A visual comparison of the denoising performance of each algorithm is provided below.

### NIDDL Dataset

| Experiment Name | Patch Size | Epochs | Mean SI_PSNR Improvement | Std SI_PSNR Improvement | Notes        |
|-----------------|------------|--------|--------------------------|-------------------------|--------------|
| NIDDL_N2V_1     | 64         | 10     | 3.227002                 | 1.755672                |              |
| NIDDL_N2V_2     | 64         | 10     | 1.844298                 | 1.395469                | Augmentation |
| NIDDL_N2V_3     | 32         | 10     | 2.504732                 | 1.618150                |              |
| NIDDL_N2V_4     | 128        | 10     | 2.944237                 | 1.854985                |              |
| NIDDL_N2V_5     | 128        | 30     | 2.995562                 | 1.818422                |              |
| NIDDL_N2V_6     | 64         | 30     | 2.779872                 | 1.703397                |              |

Noise2Void has been tested on 2D frames with different patch sizes and number of epochs. The best results were obtained with a patch size of 64, 10 epochs, and NO data augmentation. The same configuration was used as a baseline for the other experiments.

### REFINED Dataset

**NOTICE**: The results reported below are gif animations intended to provide a visual comparison of the denoising performance of each algorithm. For a detailed comparison of the results obtained with each algorithm, please refer to the respective experiment notebooks. (frame skip: 20, resolution: 10%). Previews are generated from the test video of the REFINED dataset.
Input and output may appear out of sync due to the loading time of the gif.

| Input | REFINED_N2N_1 | REFINED_N2N_1 2D N2V with no background microscope noise |
|-----------------|------------|------------|
| ![Input (test dataset)](./docs/experiments_preview/test_input.gif)  | ![REFINED_N2V_1 (test dataset)](./docs/experiments_preview/REFINED_N2N_1.gif) | ![REFINED_N2V_1_NO_AVERAGE(test dataset)](./docs/experiments_preview/REFINED_N2N_1_noavg.gif) | 
| Input | REFINED_N2V_3 (3D N2V) | REFINED_N2V_3 3D N2V with no background microscope noise |
| ![Input (test dataset)](./docs/experiments_preview/test_input.gif)  | ![REFINED_N2V_3 (test dataset)](./docs/experiments_preview/REFINED_N2V_3.gif) | ![REFINED_N2V_3_NO_AVERAGE(test dataset)](./docs/experiments_preview/REFINED_N2V_3_noavg.gif) | 









