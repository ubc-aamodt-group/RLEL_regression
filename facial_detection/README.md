
## Binary-encoded labels for facial labdmark detection


### Introduction 
This is the official code for binary-encoded labels: BEL-J and ORBC.  We evaluate our method on three datasets: COFW, 300W, WFLW, and AFLW.
We use  official code implementation provided by [1] and [2] and modify for our proposed approach. We adopt **HRNetV2-W18** for facial landmark detection.

#### Environment setup
This code is developed using on Python 3.6.12 and PyTorch 1.10.0 on Ubuntu 18.04 with NVIDIA GPUs. 

1. Install PyTorch 1.10.0 following the [official instructions](https://pytorch.org/)
2. Install dependencies
````
python -m pip install -r requirements.txt
````


#### Data

Please download images (COFW, 300W, AFLW, WFLW) from official websites and then put them into `images` folder for each dataset.

##### COFW:

Download the \*.mat from [this link](https://drive.google.com/file/d/1Z5KyYqRbymlvtQ7bqP74AgVpm7TEv2z_/view?usp=sharing) and [this link](https://drive.google.com/file/d/1ACitXQigMq7Y3x5fkoXU6eUQIOwT0AqR/view?usp=sharing). 

##### 300W: 
[part1](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.001), [part2](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.002), [part3](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.003),  [part4](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.004)
Please note that the database is simply split into 4 smaller parts for easier download. In order to create the database you have to unzip part1 (i.e., 300w.zip.001)

##### AFLW:
Please visit [the official website](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) for download instructions. 
Unzip aflw-images-{0,2,3}.tar.gz (images). 

##### WFLW:
Download the images from [this link](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view) and unzip the images. 

#### HRNetV2-W18 pretrained model

Download the ImageNet pretarined model to ``hrnetv2_pretrained`` from [this link](https://drive.google.com/file/d/1TWSQEdhGwKQrfYjIiWUvkckRVqI4tyXT/view?usp=sharing)


*download.sh provides usefule commands to download dataset WFLW/COFW/300LP, and HRNetV2-W18 ImageNet pretrained modes.*
*gdown is a python package.*


Facial_detection director structure should look like this:
````
facial_detection/
├── code_configs
│   ├── *_256_code.pkl
│   ├── *_256_tensor.pkl
├── configs
│   ├── 10_0p0003_16_WFLW_30_V1.yaml
│   ├── 10_0p0007_16_300W_30_V1.yaml
│   ├── 30_0p0005_16_AFLW_30_V1.yaml
│   └── 30_0p0005_16_COFW_30_V1.yaml
├── README.md
├── requirements.txt
├── lib
│   ├── config
│   ├── core
│   ├── datasets
│   ├── __init__.py
│   ├── models
│   └── utils
├── tools
└── data
    ├── cofw
    |   ├── COFW_test_color.mat
    |   └── COFW_train_color.mat
    ├── 300w
    │   ├── face_landmarks_300w_test.csv
    │   ├── face_landmarks_300w_train.csv
    │   ├── face_landmarks_300w_valid_challenge.csv
    │   ├── face_landmarks_300w_valid_common.csv
    │   ├── face_landmarks_300w_valid.csv
    │   └── images
    ├── aflw
    │   ├── face_landmarks_aflw_test.csv
    │   ├── face_landmarks_aflw_test_frontal.csv
    │   ├── face_landmarks_aflw_train.csv
    │   └── images
    └── wflw
        ├── face_landmarks_wflw_test_blur.csv
        ├── face_landmarks_wflw_test.csv
        ├── face_landmarks_wflw_test_expression.csv
        ├── face_landmarks_wflw_test_illumination.csv
        ├── face_landmarks_wflw_test_largepose.csv
        ├── face_landmarks_wflw_test_makeup.csv
        ├── face_landmarks_wflw_test_occlusion.csv
        ├── face_landmarks_wflw_train.csv
        └── images


````

#### Test

Run following commands to set the output directory location.
````
export TMPDIR="."
````

COFW dataset
````
python tools/test_distance.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld1.pth
python tools/test_distance.py --cfg configs/30_0p0005_16_COFW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld1_s.pth
````
300W dataset
`````
python tools/test_distance.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld2.pth
python tools/test_distance.py --cfg configs/10_0p0007_16_300W_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld2_s.pth
````
WFLW dataset
`````
python tools/test_distance.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld3.pth
python tools/test_distance.py --cfg configs/10_0p0003_16_WFLW_30_V1.yaml --suf "test" --loss "ce" --code code_configs/belu.yaml --model_file ../trained_models/fld3_s.pth
````
Training:
````
CUDA_VISIBLE_DEVICES=0 python tools/train_distance.py  --cfg configs/30_0p0005_16_COFW_30_V1.yaml  --suf fld1_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.0 --normweight 1.0 --constweight 0.0 --regweight 0.0 --distweight 3.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

CUDA_VISIBLE_DEVICES=1 python tools/train_distance.py  --cfg configs/30_0p0005_16_COFW_30_V1.yaml  --suf fld1s_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.9 --normweight 1.0 --constweight 0.0 --regweight 0.0 --distweight 4.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

CUDA_VISIBLE_DEVICES=2 python tools/train_distance.py  --cfg configs/10_0p0007_16_300W_30_V1.yaml  --suf fld2_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.0 --normweight 1.0 --constweight 1.0 --regweight 0.0 --distweight 5.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

CUDA_VISIBLE_DEVICES=3 python tools/train_distance.py  --cfg configs/10_0p0007_16_300W_30_V1.yaml  --suf fld2s_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.9 --normweight 1.0 --constweight 0.05 --regweight 0.0 --distweight 5.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

CUDA_VISIBLE_DEVICES=1 python tools/train_distance.py  --cfg configs/10_0p0003_16_WFLW_30_V1.yaml  --suf fld3_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.0 --normweight 1.0 --constweight 0.1 --regweight 0.0 --distweight 0.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

CUDA_VISIBLE_DEVICES=3 python tools/train_distance.py  --cfg configs/10_0p0003_16_WFLW_30_V1.yaml  --suf fld3s_train --code code_configs/rand256.yaml --loss smooth_ce --drop 0.9 --normweight 1.0 --constweight 0.1 --regweight 0.0 --distweight 5.0 --reqgrad 1.0 --distscale 0.0 --disttype NL1

````


## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

Official implementation: [https://github.com/HRNet/HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)
Licensed under the MIT License.

[2] D. Shah, Z. Y. Xue, and T. M. Aamodt, ``Label encoding for regression networks'', in 29th International Conference on Learning Representations, April 2022. [Online](https://openreview.net/forum?id=8WawVDdKqlL)

Official implementation: [https://github.com/ubc-aamodt-group/BEL_regression](https://github.com/ubc-aamodt-group/BEL_regression)