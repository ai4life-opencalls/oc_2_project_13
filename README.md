<p align="center">
  <a href="https://ai4life.eurobioimaging.eu/open-calls/">
    <img src="https://github.com/ai4life-opencalls/.github/blob/main/AI4Life_banner_giraffe_nodes_OC.png?raw=true">
  </a>
</p>

# Project 13: Optimizing calcium image acquisition with Machine Learning denoising algorithms

# Overview

This repository provides code for evaluating multiple denoising algorithms applied to Calcium Imaging datasets.

The primary objective of this study is to identify optimal strategies for denoising Calcium Imaging data under conditions of limited training data, high noise levels, and the absence of clean reference images. Given these constraints, this project serves as an exploratory analysis rather than an exhaustive benchmarking of denoising methodologies for Calcium Imaging.

The denoising algorithms considered in this study include Noise2Noise (N2N), Noise2Void (N2V), Hierarchical DivNoising (HDN), DeepCAD-RT, and CellMincer. We put particular emphasi on accessibility and ease of implementation, rather than denoising performances alone. To facilitate adoption by researchers, the project prioritizes algorithms with user-friendly workflows and provides detailed instructions and modifications for codebases requiring additional configuration.

For each completed experiment, we supply a set of Jupyter notebooks and standalone scripts that can be readily utilized and adapted to specific research needs.

## Getting started

Due to the computational requirements of some of the algoritms used, part of the code hosted in this repository has been designed to run as standalone scripts (e.g., on an HPC or *headless* machine).
Tutorials notebooks for the most relevant experiments are also provided. They are named as the experiment code reported in the following sections.

### Environment setup

Create a file '.env' in the current directory with the following environment variables:

```bash
    DATASET_FOLDER='your_path_to/calcium_imaging/' # Root folder of this repository
    OUTPUT_FOLDER='output/'
    MODELS_FOLDER='models/'
```

you can also specify multiple `.custom_envs` and pass them to the script using `-e .custom_env`.

# Dataset and pre-processing

In this project we used three different datasets: 
 Two of them were provided by the applicants and one is the **Mouse Neurites dataset**, [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6299076.svg)](https://doi.org/10.5281/zenodo.6299076) part of the [DeepCAD-RT](https://cabooster.github.io/DeepCAD-RT/Datasets/) repository.

## HUVEC_IMG and HUVEC_VID datasets
The dataset used in this project has been provided by researchers at McGill University in the context of AI4Life Open Challenges 2024.

The dataset consisted of HUVEC cells with GCaMP6s. 20ms exposure, 20% lamp intensity, stream acquisition with widefield imaging. 

The original dataset consisted of three folders, one named "DeepCAD", containing 30 1009x1024x1024 videos, one named "Test Data", containing 30 1009x1024x1024 videos, and one named "NIDDL", containing 163 Low-SNR and 163 High-SNR paired frames. Both DeepCAD and NIDDL are made from a combination of 3 different experiments made with the same imaging conditions. 

Please notice that the name of the folders are NOT related to the respective algorithms, but to the experiments that were performed by the applicants. To avoid confusion, we renamed the "DeepCAD" folder to **HUVEC_VID**, and the NIDDL folder to **HUVEC_IMG**. The "Test Data" folder was kept as is.

### Data Pre-processing

We used the **HUVEC_IMG** for performing some hyperparameter tuning and testing the codebase, while data from HUVEC_VID and "Test Data" was assigned for main experiments. 
Additionally, the applicants reported to have previously subtracted the average microscope background and provided a single frame of noise. Before proceeding with the experiments, we added back the background noise to retrieve RAW data and ensure consistency between the experiments.

## REFINED Dataset
As detailed in the experiment map in the following section, we discovered that most of the videos we received were affected by an issue during acquisition issue that caused the dynamic range of the images to be overly compressed (e.g., only 4-5 pixel intensities were used to express the signal). As the applicants couldn't provide corrected data, we agreed to proceed with the available data.
In particular, we proceeded to manually select videos that were less affected by this issue, resulting in the following **REFINED** dataset, (composed of only images originally from the "TestData" folder):

```
Training:
    - 20ms_20%_Yoda1_005.tif
    - 20ms_20%_Yoda1_006.tif
Validation:
    - 20ms_20%_Yoda1_008.tif
Testing:
    - 20ms_20%_Yoda1_009.tif
```

## MOUSENEU_LP Dataset

To investigate the performance of the denoising algorithms on a different dataset, we used the Mouse Neurites dataset. The dataset has been made available by the authors of DeepCAD-RT and available on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6299076.svg)](https://doi.org/10.5281/zenodo.6299076). 

Videos in this dataset consist of 3D stacks of shape (6500, 490, 490) and were acquired at 30Hz, with GCaMP6s at different power levels.  

To build the MOUSENEU_LP dataset, we selected two videos at 66mW power level, one for training and one for validation (`01_MouseNeurite_GCaMP6f_66mWpower_40umdepth_30Hz_lowSNR_MCRound1` and `02_MouseNeurite_GCaMP6f_66mWpower_70umdepth_30Hz_lowSNR_MCRound1` respectively).

# Algorithm Details

## Noise2Noise (N2N)

[Paper](https://arxiv.org/abs/1803.04189)

### Overview

Noise2Noise is a denoising framework that is based on a counterintuitive, yet powerful approach: training deep neural networks using only pairs of corrupted images, without requiring clean references. The core idea behind N2N is that when the noise is spatially uncorrelated, the expected value of a noisy observation is equal to the underlying clean signal. By minimizing a loss function - such as the L2 loss - the network is trained to predict the mean of the noisy observations, effectively denoising the data.

### Details
In practice, N2N uses a U-Net as a core architecture. The training process pairs independently corrupted images as input and target pairs, under the assumption that each image is drawn from the same distribution. 

In our experiments, due to the lack of clean reference images, we followed a similar approach to the one used in DeepCAD, were we used consecutive frames of the videos as input and target pairs during training. 

## Noise2Void (N2V)

![Paper](https://arxiv.org/abs/1811.10980)

### Overview

Noise2Void is a self-supervised denoising algorithm, based on the hypotesis that (i) the underlying structures are smooth, and (ii) the noise is pixel-wise independent. The core idea behind N2V is to train a neural network to predict the value of a pixel based on the surrounding context, without access to the pixel's true value. If the two assumptions hold, the true signal value can be estimated by the surrounding context, while the noise cannot be predicted because it is pixel-wise independent. This allows the network to learn to denoise the image by predicting the missing pixel values.

### Details
The architecture employed in N2V is based on blind-spot networks. This design excludes the central pixel from its receptive field during training. 
During training, a mask prevents the model from accessing the value of the central pixel, that the network aims to predict. Instead, it utilizes information from surrounding pixels, which encourages the model to learn how to reconstruct the central pixel's value based on its context within the image. 

In this repository, we used both 2D and 3D U-Net architectures for denoising Calcium Imaging data.

## Hierarchical DivNoising (HDN)

### Overview
Hierarchical DivNoising (HDN) is another advanced approach in image denoising that builds on concepts similar to N2V but introduces a hierarchical structure for improved performance. HDN is also capable to remove spatially-correlated noise, however this procedure requires a pre-processing step in which noise is estimated from a noisy observation and a high SNR signal.

### Details
HDN organizes the denoising process into multiple levels, allowing it to capture features at varying scales. This hierarchical approach enhances the model's ability to understand complex patterns within noisy images, improving denoising efficacy. 

In our experiments, we used the [original implementation](https://github.com/juglab/HDN) of HDN.

## DeepCAD-RT

[Paper](https://www.nature.com/articles/s41592-021-01225-0) | [Code](https://github.com/cabooster/DeepCAD-RT)

### Overview

DeepCAD is a deep learning-based denoising algorithm designed specifically for Calcium Imaging data. DeepCAD is based on the assumption that in the context of Calcium Imaging, two consecutive frames are highly correlated and can be regarded as two independent samples of the same underlying signal. This allows the model to leverage the temporal information present in the data by using frame pairs as input and target pairs during training. 

### Details

In DeepCAD, the model architecture consists of a 3D U-Net, trained with a self-supervised strategy in which the input and output are pairs of consecutive frames. In this way, the model learns to predict the next frame in the sequence based on the current frame, effectively denoising the data by exploiting the temporal correlations between frames.

## CellMincer

[Paper](https://cellmincer.readthedocs.io/en/latest/installation/index.html) | [Code](https://github.com/cellarium-ai/CellMincer/tree/master)

### Overview

CellMincer is a self-supervised framework designed to work with Voltage Imaging. designed specifically for denoising voltage imaging datasets.
CellMincer operates by masking and predicting sparse sets of pixels over short temporal windows. This allows the model to leverage both the unique information from individual frames and the context from neighboring frames. 

### Details
The architecture consists of a frame-wise 2D U-Net module for spatial feature extraction followed by a pixelwise 1D convolutional module for temporal data processing, optimizing both spatial and temporal correlations of the data.

The authors report that the performance of CellMincer on Calcium Imaging data are degraded compared to those of DeepCAD. This may be due to the different scales in temporal dynamics between Voltage and Calcium imaging. Nonetheless, this frameword remains a direction worth investigating.

-----

# Experiment Overview

The following section outlines the experiment roadmap, providing a general overview of the experimental design. This serves as a visual guide for navigating the repository, understanding the experiments conducted, and identifying potential future research directions.

For experiments executed using standalone scripts, the corresponding execution commands are documented within the respective notebooks.

The implementations of Noise2Noise (N2N) and Noise2Void (N2V) utilized in this study are based on the [CAREamics](https://careamics.github.io/) framework.


![Experiment Graph](./docs/experiments_graph.drawio.svg)

-----

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

Noise2Void has been tested on 2D frames with different patch sizes and number of epochs. We compared the difference between the [Scale Invariant PSNR](https://arxiv.org/abs/1711.00541) of LowSNR vs HighSNR and Predictions vs HighSNR. We report both mean and standard deviation across all the images in the NIDDL dataset. 

The best results were obtained with a patch size of 64, 10 epochs, and NO data augmentation. The same configuration was used as a starting baseline for the other experiments.


### Experiments on the MOUSENEU_LP Dataset

For the MOUSENEU_LP dataset, we compared the performance of Noise2Noise, Noise2Void, and DeepCAD-RT on Calcium Imaging data. The results are summarized in the table below.

| split   | target_name           | ssim          | microssim     | psnr           | si_psnr        |
|:--------|:----------------------|:--------------|:--------------|:---------------|:---------------|
| *train*   | *lowSNR*                | *0.484 ± 0.027* | *0.194 ± 0.018* | *23.698 ± 0.203* | *23.490 ± 0.538* |
| train   | **MOUSENEU_LP_DEEPCAD_1** | **0.496 ± 0.027** | **0.301 ± 0.023** | **23.752 ± 0.203** | **24.591 ± 0.576** |
| train   | MOUSENEU_LP_N2N_1     | 0.490 ± 0.026 | 0.226 ± 0.019 | 23.458 ± 0.200 | 23.973 ± 0.552 |
| train   | MOUSENEU_LP_N2V_1     | 0.489 ± 0.027 | 0.190 ± 0.016 | 23.678 ± 0.214 | 23.496 ± 0.538 |
| *val*     | *lowSNR*                | *0.411 ± 0.020* | *0.174 ± 0.014* | *20.431 ± 0.150* | *22.330 ± 0.477* |
| val     | **MOUSENEU_LP_DEEPCAD_1** | **0.421 ± 0.020** |** 0.285 ± 0.020** | **20.464 ± 0.151** | **23.713 ± 0.525** |
| val     | MOUSENEU_LP_N2N_1     | 0.415 ± 0.020 | 0.206 ± 0.016 | 20.414 ± 0.159 | 22.522 ± 0.480 |
| val     | MOUSENEU_LP_N2V_1     | 0.415 ± 0.020 | 0.186 ± 0.014 | 20.594 ± 0.159 | 22.297 ± 0.476 |

Both from quantitative analysis and visual inspection, DeepCAD-RT outperformed Noise2Noise and Noise2Void on the MOUSENEU_LP dataset.
In particular, Noise2Noise and Noise2Void fail to capture the underlying signal with the parameter used for training. DeepCAD-RT, on the other hand, was able to denoise the data effectively, - providing a cleaner output compared to the noisy input - by using the default parameters provided by the authors.

The 3D version of Noise2Void is currently being tested on the MOUSENEU_LP dataset due to high memory consumption during prediction.

### Experiments on REFINED Dataset

**NOTICE**: The results reported below are gif animations intended to provide a visual comparison of the denoising performance of each algorithm. For a detailed comparison of the results obtained with each algorithm, please refer to the respective experiment notebooks. Previews are generated from the test video of the REFINED dataset (frame skip: 20, resolution: 10%). Input and output may appear out of sync due to the loading time of the gif.

| Input | REFINED_N2N_1 | REFINED_N2N_1 2D N2V with no background microscope noise |
|-----------------|------------|------------|
| ![Input (test dataset)](./docs/experiments_preview/test_input.gif)  | ![REFINED_N2V_1 (test dataset)](./docs/experiments_preview/REFINED_N2N_1.gif) | ![REFINED_N2V_1_NO_AVERAGE(test dataset)](./docs/experiments_preview/REFINED_N2N_1_noavg.gif) | 
| Input | REFINED_N2V_3 (3D N2V) | REFINED_N2V_3 3D N2V with no background microscope noise |
| ![Input (test dataset)](./docs/experiments_preview/test_input.gif)  | ![REFINED_N2V_3 (test dataset)](./docs/experiments_preview/REFINED_N2V_3.gif) | ![REFINED_N2V_3_NO_AVERAGE(test dataset)](./docs/experiments_preview/REFINED_N2V_3_noavg.gif) | 

By performing a visual inspection of the results we can observe that the N2N and the N2V algorithms were able to denoise the data effectively, in contrast to the results obtained on the MOUSENEU_LP dataset. DeepCAD-RT still produce good results. A qualitative comparison of the results shows that N2N tends to produce smoother outputs with respect to the others. When comparing 3D Noise2Void with DeepCAD-RT, we can observe that both algorithms were able to denoise the data effectively, with Noise2Void producing an output with slighly more contrast and level of detail. However, due to the lack of a clean reference, it is difficult to assess the fidelity of the denoised data to the original signal.

### Discussion

As stated in the previous sections, the main objective of this project was to identify optimal strategies for denoising Calcium Imaging data under conditions of limited training data, high noise levels, and the absence of clean reference images. The focus of this study was on the accessibility and ease of implementation of the denoising algorithms, rather than their denoising performances alone.

#### User-friendliness and ease of implementation
In our experiments, we also planned to include more advanced denoising algorithms such as Hierarchical DivNoising and CellMincer. However, we had to face some issues during the implementation of these algorithms, which prevented us from including them in the final analysis and forced us to focus on the algorithms that were easier to implement. During our efforts in implementing the codebases, we allocated a limited amount of time to fix eventual issues that arose for each algorithm. However, our aim was to provide a comprehensive overview of the denoising algorithms with minimal modifications to the original codebases.
Specifically, we encountered difficulties in setting up the Hierarchical DivNoising (HDN) algorithm due to runtime errors that affected the fitting of noise models with this kind of data. After attempting to fix the issues, we found the training of the model to be numerically unstable for this dataset. We contacted the authors of CAREamics to request assistance in resolving the issue and were informed that the developers are currently working to enable the future implementation of HDN in their codebase. As a result, we were unable to include HDN in the current analysis, but we envision that this results may be produced as soon as the model is available. For CellMincer, we encountered issues with the import of the package, which initially prevented us from running the code. After trying to reach the authors for support on GitHub, we were able to solve the issue and run the code by patching the package manually. However, after running the code, we found that CellMincer was not able to produce denoised outputs for our REFINED dataset. While this is in line with the authors' report that CellMincer may not always perform well on Calcium Imaging data, we are confident that with further modifications to the model architecture, it could represent a valuable tool for denoising Calcium Imaging data. Algorithms in the CAREamics framework (i.e., N2N and N2V) have proven to be generally easy to train and use also on desktop computers, however they sometimes required an unexpectedly high amount of RAM during the prediction phase, especially on the MOUSENEU_LP dataset or with 3D models. This may be a limitation for researchers with limited computational resources. CAREamics authors are aware of this issue and are working on a solution to mitigate this problem. Lastly, we found that the DeepCAD-RT algorithm was the most user-friendly and easy to implement, providing good denoising performances on both the MOUSENEU_LP and REFINED datasets. DeepCAD-RT is in fact an iterative improvement of the original DeepCAD algorithm, meant to provide real-time denoising capabilities for Calcium Imaging data. 

#### Denoising performances
This exploratory analysis has provided valuable insights into the performance of different denoising algorithms on Calcium Imaging data. While this study is not exhaustive, it highlights the importance of tailoring denoising algorithms to the specific characteristics of the data. In particular, we used two different datasets to evaluate the performance of Noise2Noise, Noise2Void, and DeepCAD-RT on Calcium Imaging data. The results show that DeepCAD-RT provided good performances both on the MOUSENEU_LP and REFINED datasets, while Noise2Noise and Noise2Void struggled to capture the underlying signal in the former dataset. In the latter dataset, all algorithms were able to denoise the data effectively, with Noise2Noise producing smoother outputs compared to Noise2Void and DeepCAD-RT. One possible explaination for the difference in performance between the two datasets if the higher noise levels present in the MOUSENEU_LP dataset, which may have made it more challenging for Noise2Noise and Noise2Void to denoise the data effectively. In contrast, the REFINED dataset had a better Signal-to-Noise ratio, while being more affected by correlated noise. Moreover, for the REFINED dataset, we found that 3D Noise2Void seems to have a slight edge over DeepCAD-RT in terms of image quality, and this highlights the need for further investigation into the performance of 3D denoising algorithms on Calcium Imaging data.


### Conclusions
In conclusion, this exploratory analysis has provided valuable insights into the performance of various denoising algorithms on Calcium Imaging data. While DeepCAD-RT demonstrated robust performance across different datasets, Noise2Noise and Noise2Void showed varying levels of effectiveness depending on the dataset characteristics. The study underscores the importance of tailoring denoising algorithms to the specific properties of the data, such as noise levels and signal-to-noise ratios.

The challenges encountered in implementing some advanced algorithms highlight the need for further development and refinement to enhance their usability and performance. Additionally, the promising results obtained with 3D Noise2Void suggest that further investigation into 3D denoising techniques could yield significant improvements in image quality. More generally, this project emphasizes the need for research and development in the field of Calcium Imaging denoising, particularly under conditions of limited training data and high noise levels. We want to highlight the importance of collaboration between researchers and developers to address the challenges associated with implementing advanced denoising algorithms and to facilitate the adoption of these tools by the scientific community.

### Bibliography

Mouse Neurite dataset is from the DeepCAD-RT Paper:
- Xinyang Li, Yixin Li, Yiliang Zhou, et al. Real-time denoising enables high-sensitivity fluorescence time-lapse imaging beyond the shot-noise limit. Nat. Biotechnol. (2022). https://doi.org/10.1038/s41587-022-01450-8
- Xinyang Li, Guoxun Zhang, Jiamin Wu, et al. Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. Nat. Methods 18, 1395–1400 (2021). https://doi.org/10.1038/s41592-021-01225-0

