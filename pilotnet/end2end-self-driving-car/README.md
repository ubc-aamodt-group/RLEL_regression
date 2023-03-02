# Binary-encoded labels for end-to-end autonomous driving

## Introduction

This is the official code for binary-encoded labels on Sully Chen's Driving Dataset. We develop an implementation based on [1] and [2] and use PilotNet as our network architecture. 

# Environment setup
This code is developed using Python 3.8.3 and PyTorch 1.10.0 on Ubuntu 18.04 with NVIDIA RTX 2080 Ti GPUs.

## Conda
Create the conda environment by:

```
conda env create -f environment.yml
```

# Data

The dataset is provided by Sully Chen at [3]
Please use the provided `download_dataset.sh` script to download the data and unzip it into end2end-self-driving-car/driving_dataset

We provide train/test splits for the dataset in `train.csv` and `val.csv`. 

Your directory structure should look like:

```
pilotnet
 ┣ end2end-self-driving-car
 ┃ ┣ ckpt/
 ┃ ┣ config
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ defaults.py
 ┃ ┣ data
 ┃ ┃ ┣ datasets
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ baladmobile.py
 ┃ ┃ ┃ ┗ driving_data.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┗ build.py
 ┃ ┣ driving_dataset
 ┃ ┃ ┣ .DS_Store
 ┃ ┃ ┣ data.txt
 ┃ ┃ ┣ train.csv
 ┃ ┃ ┣ val.csv
 ┃ ┃ ┗ vis.csv
 ┃ ┣ encodings/
 ┃ ┣ model
 ┃ ┃ ┣ engine
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ evaluation.py
 ┃ ┃ ┃ ┣ pruning.py
 ┃ ┃ ┃ ┣ trainer.py
 ┃ ┃ ┃ ┗ visualization.py
 ┃ ┃ ┣ layer
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ feed_forward.py
 ┃ ┃ ┣ meta_arch
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ backward_pilot_net.py
 ┃ ┃ ┃ ┣ pilot_net.py
 ┃ ┃ ┃ ┗ pilot_net_analytical.py
 ┃ ┃ ┣ solver
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ build.py
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ build.py
 ┃ ┃ ┣ conversion_helper.py
 ┃ ┃ ┗ transforms.py
 ┃ ┣ util
 ┃ ┃ ┣ __init__.py
 ┃ ┃ ┣ logger.py
 ┃ ┃ ┣ prepare_driving_dataset.py
 ┃ ┃ ┣ save_image.py
 ┃ ┃ ┗ visdom_plots.py
 ┃ ┣ .gitignore
 ┃ ┣ drive.py
 ┃ ┣ get_min_max.py
 ┃ ┗ main.py
 ┣ .gitignore
 ┣ README.md
 ┣ download_dataset.sh
 ┗ environment.yml
```

# Testing

```
python main.py --mode test --transform u --reverse-transform cor --init rand --loss smooth_ce  MODEL.WEIGHTS ../../trained_models/PN_trained.pth.tar
```

# Training

If you would like to train our models, please use the command below. 
```
CUDA_VISIBLE_DEVICES=0 python main.py --transform u --reverse-transform cor --loss smooth_ce --reqgrad 1.0 --dist_weight 0.0 --const_weight 2.0 --init rand --version V1
```

# References
[1] M. Fathi, MahanFathi/end2end-self-driving-car. 2020. Available: https://github.com/MahanFathi/end2end-self-driving-car

[2] D. Shah, Z. Y. Xue, and T. M. Aamodt, ``Label encoding for regression networks'', in 29th International Conference on Learning Representations, April 2022. [Online](https://openreview.net/forum?id=8WawVDdKqlL)

Official implementation: [https://github.com/ubc-aamodt-group/BEL_regression](https://github.com/ubc-aamodt-group/BEL_regression)

[3]S. Chen, SullyChen/driving-datasets. 2021. Available: https://github.com/SullyChen/driving-datasets
