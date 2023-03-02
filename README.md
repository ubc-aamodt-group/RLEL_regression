# [Learning Label Encodings for Deep Regression <img src="images/iclr-logo.png" width=200>](https://openreview.net/pdf?id=k60XE_b0Ix6)

[Deval Shah](https://www.linkedin.com/in/deval-shah-91485867/),, [Tor M. Aamodt](https://www.ece.ubc.ca/~aamodt/)

This repository contains code for the work on "[Learning Label Encodings for Deep Regression](https://openreview.net/pdf?id=k60XE_b0Ix6)"  to be presented in the ICLR 2023 (**Spotlight presentation**). 



Table of Contents
=================

[Head pose estimation with ResNet50](hpe_resnet50): Training and inference code for head pose estimation.

[Facial landmark detection with HRNetV20W18](facial_detection): Training and inference code for facial landmark detection.

[Age estimation with ResNet50](age_estimation): Training and inference code for age estimation.

[End-to-end autonomous driving with PilotNet](pilotnet): Training and inference code for end-to-end autonomous driving with PilotNet feature extractor. 
Trained models can be downloaded from [https://drive.google.com/file/d/1mWZiNyXcUrwLvCetdtQykhMj2qUmy7Fa/view?usp=sharing](https://drive.google.com/file/d/1mWZiNyXcUrwLvCetdtQykhMj2qUmy7Fa/view?usp=sharing)
Download the models to trained_models/ directory. 
You can use the following commands to download trained models.

mkdir trained_models
cd trained_models
gdown https://drive.google.com/uc?id=1mWZiNyXcUrwLvCetdtQykhMj2qUmy7Fa
unzip trained_models.zip
cd ../

## Citation

If you find this project useful in your research, please cite:

```
```
@inproceedings{ShahICLR2023,
  author    = {Shah, Deval and Aamodt, Tor M. },
  booktitle = {International Conference on Learning Representations},
  title     = {Learning Label Encodings for Deep Regression},
  url = {https://openreview.net/pdf?id=k60XE_b0Ix6},
  month     = {May},
  year      = {2023},
}
```
```
