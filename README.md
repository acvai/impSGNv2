# ImpSGNv2: Improved Semantic-Guided Networkwith Attention-based Graph Convolution (GCNs)For Skeleton-based Action Recognition

## Introduction

A critical challenge in skeleton-based action recognition lies in effectively extracting meaningful features from skeleton joints. However, existing state-of-the-art networks for this task are often complex and heavily parameterized, leading to inefficient training and inference times, particularly on large-scale datasets. In this work, we propose a simple yet efficient baseline for skeleton-based Human Action Recognition (HAR), building on our previous model, ImpSGN. Our new approach features a lighter and faster architecture while preserving the same performance capabilities. The proposed architecture employs attention-based graph convolution to effectively capture the intricate interconnections within skeletal structures. We report competitive accuracy on the large-scale NTU-RGB+D 60 dataset, achieving 90.0% and 95.0% on the Cross-Subject and Cross-View benchmarks, respectively. On NTU-RGB+D 120, the model achieves 84.7% and 85.8% on the Cross-Subject and Cross-Setup benchmarks, respectively.
This work demonstrates a significant improvement over our previous model, ImpSGN, by extracting more discriminative spatial and temporal features, all while reducing the number of parameters by 60\%
 

<div align=center>
<img src="https://github.com/acvai/impSGNv2/blob/master/images/SOTA_comparison.PNG" width = 50% height = 50% div align=center>
</div>

Figure 1: The Cross-Subject (CS) benchmark of NTU-RGB+D 60 evaluates accuracy and network size (number of learning parameters). ImpSGNv2 excels with strong performance and a compact design.



## Framework
![image](https://github.com/acvai/impSGNv2/blob/master/images/ImpSGNv2.PNG)

Figure 2: Framework of the proposed end-to-end ImpSGNv2 model, consisting of a single stream with multiple SAT blocks. TheEmbed module encodes joint dynamics by integrating position (Pos), velocity (Vel), and bone features (Bone). Each SAT Blocksequentially applies a Spatial-with-self-Attention module followed by a Temporal module.

## Prerequisites
The code is built with the following libraries:
- Python 3.7
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/) 1.13.1

Check my environment.yml file that contains the necessary pacakges (version for my own machine). You may use '''conda create -n impSGNv2 -f environment.yml'''

## Data Preparation

We use the dataset of NTU60 RGB+D as an example for description. We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset.

- Extract the dataset to ./data/ntu/nturgb+d_skeletons/
- Process the data
```bash
 cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


## Training

```bash
# For the CS setting
python  main_ImpSGNv2.py --network ImpSGNv2 --train 1 --case 0
# For the CV setting
python  main_ImpSGNv2.py --network ImpSGNv2 --train 1 --case 1
```

## Testing

- Test the pre-trained models (./results/NTU/ImpSGNv2/)
```bash
# For the CS setting
python  main_ImpSGNv2.py --network ImpSGNv2 --train 0 --case 0
# For the CV setting
python  main_ImpSGNv2.py --network ImpSGNv2 --train 0 --case 1
```

## Reference

This repository holds the code for the following paper:

[ImpSGNv2: Improved Semantic-Guided Networkwith Attention-based Graph Convolution (GCNs)For Skeleton-based Action Recognition](https://ieeexplore.ieee.org/document/11099308/metrics#metrics). IEEE Conference.

If you find our paper and repo useful, please cite our paper. Thanks!

```
@INPROCEEDINGS{11099308,
  author={Mansouri, Amine and Elzaar, Abdellah and Bakir, Toufik},
  booktitle={2025 International Conference on Control, Automation and Diagnosis (ICCAD)}, 
  title={ImpSGNv2: Improved Semantic-Guided Network with Attention-based Graph Convolution (GCNs) For Skeleton-based Action Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Training;Convolution;Graph convolutional networks;Architecture;Semantics;Benchmark testing;Feature extraction;Skeleton;Human activity recognition;Indexes;Deep Learning;Human Action Recognition (HAR);Convolutional Neural Networks (CNNs);Graph Convolutional Networks (GCNs);Attention mechanism},
  doi={10.1109/ICCAD64771.2025.11099308}}


```
## ContrAcknoledgment

This project is inspired from our previous model impSGN [https://www.sciencedirect.com/science/article/pii/S1047320324002372](https://www.sciencedirect.com/science/article/pii/S1047320324002372) that in tern used heavily the SGN model code [Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition (SGN)](https://github.com/microsoft/SGN)

