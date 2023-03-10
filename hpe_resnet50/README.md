## Binary-encoded labels for head pose estimation


### Introduction 
This is the official code for binary-encoded labels.  We evaluate our method on three datasets: 300LP/AFLW2000 and BIWI. This code uses ResNet50 feature extractor. 
We use  official code for HopeNet [1] and BEL [2] and modify for our proposed approach BEL. 

#### Environment setup
This code is developed using on Python 3.6 and PyTorch 1.10.0 on Ubuntu 18.04 with NVIDIA GPUs. 

1. Create a virtual environment and activate with python==3.6.12 (**optional to use virtual environment) 
2. Install dependencies
````
python -m pip install -r requirements.txt
````

#### Data

1. You need to download images 300W_LP, AFLW2000, and BIWI from official websites and then put them into `datasets/*` folder for each dataset. 

Useful links:
300W_LP: [Download link](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing)
AFLW2000: [Download link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip)
BIWI: [Contact information](https://icu.ee.ethz.ch/research/datsets.html)


*download.sh provides usefule commands to download the trained models and dataset AFLW2000.*
*gdown is a python package.*

Your `datasets` and `trained_model` directories should look like this:

````
Datasets/
├── 300W_LP
│   ├── AFW
│   ├── AFW_Flip
│   ├── HELEN
│   ├── HELEN_Flip
│   ├── IBUG
│   ├── IBUG_Flip
│   ├── landmarks
│   │   ├── AFW
│   │   ├── HELEN
│   │   ├── IBUG
│   │   └── LFPW
│   ├── LFPW
│   ├── LFPW_Flip
│   ├── overall_filelist_clean.txt
│   ├── run_crop_300lp.sh
│   └── crop_300lp.py
├── AFLW2000
│   ├── imgx.jpg
│   ├── imgx.mat
│   ├── crop_AFLW2000.py
│   ├── overall_filelist_clean.txt
│   └── run_crop_AFLW2000.sh
└── BIWI
    ├── 30_test.txt
    ├── 70_train.txt
    └── hpdb
        ├── 01
	:  
        └── 24
````


#### Pre-process dataset
We pre process 300W_LP and AFLW2000 datasets before training/testing and loosely crop around the center. The scripts for preprocessing the images are taken from the official code implementation of FSA-Net [3]. 

##### 300W_LP 
````
cd 300W_LP
bash run_crop_300lp.sh
````
##### AFLW2000 
````
cd AFLW2000
bash run_crop_aflw2000.sh
````  

Run the following command to set the output directory:
```
export TMPDIR="./
```
#### Test
Please refer to the arguments options in resnet_test.py (use run_test.sh to run all the tests)

Protocol 1:
````
python BIWI/resnet_test.py --data_dir datasets/biwi/hpdb/ --filename_list datasets/biwi/70_train.txt --test_filename_list datasets/biwi/30_test.txt --output_string test --dataset BIWI --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --model_file ../trained_models/LFH1_trained.pth  --code u --code_bits 150 --loss smooth_bce
````
Protocol 2:
````
python 300LP_AFLW/resnet_test.py --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/overall_filelist_clean.txt --test_filename_list datasets/AFLW2000/overall_filelist_clean.txt --output_string test --dataset Pose_300W_LP_random_ds --batch_size 16 --lr 0.00001 --num_epochs 20 --arch resnet --num_bits 10 --gpu 0 --model_file ../trained_models/LFH2_trained.pth --code u --code_bits 200 --loss smooth_ce
````
#### Training
Please refer to the arguments options in resnet_train.py


Protocol 1:
````
python BIWI/resnet_train.py --data_dir datasets/biwi/hpdb/ --filename_list datasets/biwi/70_train.txt --test_filename_list datasets/biwi/30_test.txt --output_string smooth_ce_V1 --dataset BIWI --batch_size 8 --lr 0.0001 --num_epochs 50 --arch resnet --num_bits 10 --gpu 0 --code u --code_bits 150 --loss smooth_ce  --init rand --reqgrad 1.0 --norm_weight 1.0 --dist_weight 0.5 --const_weight 1.0
````
Protocol 2:
````
CUDA_VISIBLE_DEVICES=1 python 300LP_AFLW/resnet_train.py --data_dir datasets/300W_LP/ --filename_list datasets/300W_LP/overall_filelist_clean.txt --test_filename_list datasets/AFLW2000/overall_filelist_clean.txt --output_string 300lp_ce_V1 --dataset Pose_300W_LP_random_ds --batch_size 16 --lr 0.00001 --num_epochs 20 --arch resnet --num_bits 10 --gpu 0 --code u --code_bits 200  --loss smooth_ce  --init rand --reqgrad 1.0 --norm_weight 1.0 --dist_weight 2.0 --const_weight 0.0
````
Modify the code and loss arguments to use different encoding and loss functions. 
 
## Reference

[1] Ruiz, N., Chong, E., & Rehg, J. M. (2018). Fine-grained head pose estimation without keypoints. _IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops_, _2018_-_June_, 2155–2164. https://doi.org/10.1109/CVPRW.2018.00281 
Official implementation:  [https://github.com/natanielruiz/deep-head-pose](https://github.com/natanielruiz/deep-head-pose)

[2] D. Shah, Z. Y. Xue, and T. M. Aamodt, ``Label encoding for regression networks'', in 29th International Conference on Learning Representations, April 2022. [Online](https://openreview.net/forum?id=8WawVDdKqlL)

Official implementation: [https://github.com/ubc-aamodt-group/BEL_regression](https://github.com/ubc-aamodt-group/BEL_regression)

[3] Yang, T. Y., Chen, Y. T., Lin, Y. Y., & Chuang, Y. Y. (2019). Fsa-net: Learning fine-grained structure aggregation for head pose estimation from a single image. _Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition_, _2019_-_June_, 1087–1096. https://doi.org/10.1109/CVPR.2019.00118
Official implementation:  [https://github.com/shamangary/FSA-Net](https://github.com/shamangary/FSA-Net)
