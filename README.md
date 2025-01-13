# CalciumImagingDenoising
This repository contains code for running multiple denoising algorithms on a Calcium Imaging dataset.

## Getting started

Due to the computational requirements of some of the algoritms used, part of the code hosted in this repository has been designed to run on HPC.
Tutorials notebooks for the most relevant experiments are also provided. They are named as the experiment code. 
A detailed map for the experiments is provided in the following section.

### Setting Environment

Create a file '.env' in the current directory with the following environment variables:

```bash
    DATASET_FOLDER='your_path_to/calcium_imaging/' # Root folder of this repository
    OUTPUT_FOLDER='output/'
    MODELS_FOLDER='models/'
```

you can also specify multiple `.custom_envs` and pass them to the script using `-e .custom_env`.

# Experiment Overview

Here follows the experiment map with the general overview of the experiment design.

![Experiment Graph](./docs/experiments_graph.drawio.svg)


# Algorithm Details

## Noise2Void (N2V)

### Overview

Noise2Void operates under a self-supervised learning paradigm, exploiting the statistical properties of noise and signal in images. The core assumption is that while the signal (the denoising target) is not pixel-wise independent, the noise is conditionally independent given the signal. This allows N2V to extract meaningful information from a single noisy image by focusing on the relationship between neighboring pixels rather than relying on an exact match with a clean target.

### Details
The architecture employed in N2V is based on blind-spot networks. This design excludes the central pixel from its receptive field during training. 
During training, a mask prevents the model from accessing the value of the central pixel, that the network aims to predict. Instead, it utilizes information from surrounding pixels, which encourages the model to learn how to reconstruct the central pixel's value based on its context within the image.

### Advantages
One of the most significant advantages of Noise2Void is that it doesn't require ground truth images or high SNR images to be trained. This characteristic makes N2V particularly suitable for denoising microscopy images, where obtaining clean reference images can be challenging.

Moreover, N2V has been shown to perform competitively against other denoising methods that do have access to clean targets or noisy pairs, despite its more limited training data requirements.

## Hierarchical DivNoising (HDN)

### Overview
Hierarchical DivNoising (HDN) is another advanced approach in image denoising that builds on concepts similar to N2V but introduces a hierarchical structure for improved performance. HDN is also capable to remove spatially-correlated noise, however this procedure requires a pre-processing step in which noise is estimated from a noisy observation and a high SNR signal.

### Details
HDN organizes the denoising process into multiple levels, allowing it to capture features at varying scales. This hierarchical approach enhances the model's ability to understand complex patterns within noisy images, improving denoising efficacy. Moreover, 

### Advantages
The hierarchical structure of HDN facilitates better contextual understanding and feature extraction compared to traditional methods. This can lead to better results in scenarios with significant noise levels or complex image structures, making HDN a valuable tool in applications requiring high-quality image restoration.






