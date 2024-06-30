## Reinforcement Learning-based Layer-wise Aggregation for Personalized Federated Learning

This repo is the PyTorch implementation of pFedRLLA.


### Environment
First, clone the repository locally:
```
git clone https://github.com/fdksd/pFedRLLA.git
```
Second, install [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive).

Finally, install conda and create conda env.
```
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match CUDA version
```

### Dataset & Model
[PFLlib](https://github.com/TsingZ0/PFLlib?tab=readme-ov-file) provides a wealth of models and datasets, but in our paper, the datasets we use are CIFAR-10, CIFAR-100, Omniglot, MNIST*.
(The MNIST* dataset was constructed by us. The dataset is a hybrid dataset constructed from three datasets (MNIST, MNIST-M, and USPS),  and in this case, we established three distinct client groups, with each group containing different domain samples.)

The model uses 6-layer CNN, and details of the model can be found in `system/flcore/trainmodel/models.py`.

### Dataset Distribution
In our experiment, we used multiple heterogeneous scenarios. The specific implementation method can be seen in [PFLlib](https://github.com/TsingZ0/PFLlib/tree/master).
- Pathological non-IID.
- Practical non-IID (Dirichlet non-IID in the paper).
- real-world scenario (Omniglot).

### How to start simulating (examples for our algorithm)
```
# using the CIFAR-10 dataset, our proposed algorithm, and the 6-layer CNN model
cd ./system
python -u main.py -lbs 64 -ls 5 -lr 0.01 -nc 100 -jr 0.1 -nb 10 -data Cifar10_u100_s5 -m cnn -gr 500 -did 0 -algo FedDRL
```
We have provided the script file in `system/scripts`, which contains the hyperparameter information of each algorithm in our experiment.

> Our code is in `serverours.py` and `clientours.py`
