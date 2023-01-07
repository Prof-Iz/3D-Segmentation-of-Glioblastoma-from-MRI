# 3D-Segmentation-of-Glioblastoma-from-MRI

Repository for Final Year Project by Ibrahim Izdhan for the fulfillment of BEng (Hons) Electrical and Electronics Engineering Degree


## File Structure
---

1. Glioblastoma Segmentation Notebook
    - Working File for Training and Model Inference
1. Models Folder
    - Folder to store weights of trained models and other backbones
1. Utilities Folder
    - Helper functions such as preapration of train, validation and test data sets


## Setting up
---

### Environment
The environment is based on `python 3.9`. To create the environment in [Anaconda](https://www.anaconda.com/products/distribution) by using the following command once the repository has been pulled.

```shell
conda env create --file environment.yaml
```

This would create an environment called `fyp`.

### Dataset
The Dataset was sourced from the BraTS 2020 dataset. Search for it online and ensure that it is in the `niftii` format (MRI Scan).