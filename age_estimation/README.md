# Binary encoded labels for age estimation

## Introduction
This is the official code for binary-encoded labels on the AFAD and MORPH-II datasets. We develop our own implementation based on [1]. We adopt ResNet-50 for age estimation. 

# Environment setup
This code is developed using Python 3.8.3 and PyTorch 1.10.0 on Ubuntu 18.04 with NVIDIA RTX 2080 Ti GPUs.

## Conda
Create the conda environment by:
```
conda env create -f environment.yml
```
Install the age estimation module:
```
pip install -e .
```

# Data
Please download images (AFAD, MORPH-II). We provide the train/valid/test splits used in our experiments in the `dataset` directory

## AFAD

Download AFAD dataset from [this link][https://afad-dataset.github.io/]

Unzip the images into the appropriate folder, then modify `train_main_afad.py` with the base directory. 

## MORPH-II
Acquire the MORPH-II dataset from [this link][https://ebill.uncw.edu/C20231_ustores/web/store_main.jsp?STOREID=4]

Unzip the images into the appropriate folder and run `preprocess-morph2.py` on that folder. 

Final structure should look like:

```
age_estimation
 ┣ age_estimation
 ┃ ┣ train.py
 ┃ ┣ train_main_afad.py
 ┃ ┣ train_main_iw.py
 ┃ ┗ train_main_morph2.py
 ┣ ageresnet
 ┃ ┣ data
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ afad.py
 ┃ ┃ ┣ imdb_wiki.py
 ┃ ┃ ┣ logger.py
 ┃ ┃ ┣ morph2.py
 ┃ ┃ ┣ record.py
 ┃ ┃ ┗ transforms.py
 ┃ ┣ models
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ age_resnet.py
 ┃ ┃ ┣ age_resnet_quant.py
 ┃ ┃ ┣ pruning.py
 ┃ ┃ ┗ quant_layers.py
 ┃ ┣ utils
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ function.py
 ┃ ┗ __init__.py
 ┣ dataset
 ┃ ┣ MORPH2-preprocess
 ┃ ┃ ┗ preprocess_morph2.py
 ┃ ┣ AFAD-FULL/
 ┃ ┣ afad_test.csv
 ┃ ┣ afad_train.csv
 ┃ ┣ afad_valid.csv
 ┃ ┣ morph2-aligned/
 ┃ ┣ morph2_test.csv
 ┃ ┣ morph2_train.csv
 ┃ ┣ morph2_valid.csv
 ┃ ┗ wiki.csv
 ┣ encodings/
 ┣ ckpt/
 ┣ README.md
 ┣ environment.yml
 ┣ requirements.txt
 ┗ setup.py
```
# Inference
We can run inference via: 
```
CUDA_VISIBLE_DEVICES=0 python train.py  --transform u --dataset morph2 --gpus 0 --loss smooth_ce --reverse-transform ex --init rand --ckpt ../../trained_models/AE1_trained.pth.tar  --mode test

CUDA_VISIBLE_DEVICES=0 python train.py  --transform u --dataset afad --gpus 0 --loss smooth_ce --reverse-transform ex --init rand --ckpt ../../trained_models/AE2_trained.pth.tar  --mode test
```

More details are described in the code.

# Training
```
CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset morph2 --gpus 0 --mode train --reverse-transform ex --loss smooth_ce --dist_weight 0.0 --const_weight 2.0 --init rand --initrand rand --version VF --reqgrad 1.0

CUDA_VISIBLE_DEVICES=0 python train.py --num-epochs 50 --transform u --dataset afad --gpus 0 --mode train --reverse-transform ex --loss smooth_ce --dist_weight 0.0 --const_weight 5.0 --init rand --initrand rand --version VF --reqgrad 1.0
```

[1] D. Shah, Z. Y. Xue, and T. M. Aamodt, ``Label encoding for regression networks'', in 29th International Conference on Learning Representations, April 2022. [Online](https://openreview.net/forum?id=8WawVDdKqlL)

Official implementation: [https://github.com/ubc-aamodt-group/BEL_regression](https://github.com/ubc-aamodt-group/BEL_regression)
