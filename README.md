# Synergistic Dual Proxies: Enhancing Cohesion and Separability in Deep Graph Clustering

This repository contains the code for the paper "Synergistic Dual Proxies: Enhancing Cohesion and Separability in Deep Graph Clustering".

## Introduction
This work proposes DP-Net, an end-to-end graph clustering framework. DP-Net synergistically leverages Internal Cluster Centers (ICCs) to enforce intra‐cluster cohesion and External Cluster Consultants (ECCs) to explicitly delineate inter‐cluster boundaries. By integrating a novel dual-proxy clustering loss with multi-view consistency regularization, DP-Net effectively balances attracting nodes toward cluster cores while repelling them from ambiguous boundary regions, all within a scalable optimization scheme. This repository provides the code to reproduce the experiments presented in the paper.

## Requirements
To set up the environment and dependencies for running the code, please follow the instructions below. It is recommended to use `conda` for managing the Python environment.

### Environment
- **Python** (tested on 3.8): We recommend using the conda package manager.
  ```sh
  conda create -n dp_net python=3.8
  conda activate dp_net
  ```

### Dependencies
- **PyTorch** (tested on 1.8.1+cu111): Install PyTorch with CUDA support (CPU version is also supported). Please see the [official PyTorch website](https://pytorch.org/) for details.
  ```sh
  pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1
  ```

- **PyTorch Geometric** (tested on 2.1.0.post1): Please refer to the [official PyTorch Geometric website](https://pytorch-geometric.readthedocs.io/) for installation details.
  > If you encounter errors while importing `torch_sparse`, please re-install it with `torch-sparse==0.6.12`.
  ```sh
  conda install pyg -c pyg
  ```

Other dependencies are listed in `requirements.txt` and can be installed using:
```sh
pip install -r requirements.txt
```

## Datasets
The datasets utilized in this project will be automatically downloaded and processed upon initial execution. Furthermore, these datasets are publicly available and are not subject to licensing restrictions.

## Running the Code
To reproduce the experimental results on the Cora dataset (tested on a single NVIDIA GeForce RTX 3090 GPU (24GB)), execute the following command:
```sh
python main_cora.py
```
For more details on the setup and reproducibility on other datasets, please refer to our paper.
