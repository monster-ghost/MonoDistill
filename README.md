# MonoDistill: Learning Spatial Features for Monocular 3D Object Detection

By Zhiyu Chong, [Xinzhu Ma](https://scholar.google.com/citations?user=8PuKa_8AAAAJ), Hong Zhang, Yuxin Yue, [Haojie Li](https://scholar.google.com/citations?user=pMnlgVMAAAAJ), [Zhihui Wang](https://scholar.google.com/citations?hl=zh-CN&user=Ht19Pc0AAAAJ), [Wanli Ouyang](https://wlouyang.github.io/).

## Introduction
In this work, we propose the [MonoDistill](https://arxiv.org/pdf/2201.10830.pdf), which introduces spatial cues to the monocular 3D detector based on the knowledge distillation mechanism. Compared with previous schemes, which share
the same motivation, our method avoids any modifications on the target model and directly learns
the spatial features from the model rich in these features. This design makes the proposed method
perform well in both performance and efficiency.

## Usage

### Installation
This repo is tested on our local environment (python=3.7, cuda=9.0, pytorch=1.1), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n mono3d python=3.7
```
Then, activate the environment:
```bash
conda activate mono3d
```

Install  Install PyTorch:

```bash
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
```

and other  requirements: 
```bash
pip install -r requirements.txt
```


# Getting Started
## Training and Inference
* Download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

* Download the precomputed [depth maps]((https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing)) for the KITTI training set, which are provided by [CaDDN](https://github.com/TRAILab/CaDDN).
```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |depth_2/
          |label_2/
        |testing/
          |calib/
          |image_2/
```
* Download the [RGB Net pretrain model](https://drive.google.com/file/d/1MhyfcY3GBdNkJRU3UrzKtRWVDLw86_c9/view?usp=sharing). In our experiment, performance will degrade due to modal differences if you train from scratch. We recommend you load the pre-train model. 
  
    |                                             | Easy@R40 | Moderate@R40 | Hard@R40  |
    |---------------------------------------------|:-------:|:-------:|:-------:|
    | Baseline | 18.43 | 14.62 | 12.45 |
    
    Or you can also train baseline model by the following command：
    ```bash
    cd experiments/example
    CUDA_VISIBLE_DEVICES=0,1 python ../../tools/train_val.py --config kitti_example_centernet.yaml
    ```

* Download the [LiDAR Net pretrain model](https://drive.google.com/file/d/1UtBsm9GonhQStu9ew_JkTy6xenDAuTzF/view?usp=sharing). You can also train LiDAR Net by yourself and just change the input the network. 

    |                                             | Easy@R40 | Moderate@R40 | Hard@R40  |
    |---------------------------------------------|:-------:|:-------:|:-------:|
    | LiDAR Net | 60.57 | 45.06 | 37.90 |



* After loading RGB model and LiDAR model, you can distill it by the following command:
    ```bash
    cd experiments/example
    CUDA_VISIBLE_DEVICES=0,1 python ../../tools/train_val.py --config kitti_example_distill.yaml
    ```

## Pre-trained checkpoints
|                                             | Easy@R40 | Moderate@R40 | Hard@R40  | Link  |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|
| MonoDistill | 24.40 | 18.47 | 16.46 | [checkpoints](https://drive.google.com/file/d/1v6bje33Itxq9UoUnuAz30SzlpzfziLqA/view?usp=sharing) |



## Acknowlegment
This repo benefits from the excellent work [MonoDLE](https://github.com/xinzhuma/monodle) and [reviewKD](https://github.com/dvlab-research/ReviewKD). Please also consider citing it.




