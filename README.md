# Introduction to Machine learning

This repository contains material for introducing common approaches in the machine learning pipeline. 
It consists of several jupyter notebooks for various tasks and introduces privacy-preserving machine learning
using encrypted inference. Finally, the concept of Membership Inference Attack (MIA), which can lead to
a serious privacy leakage is introduced.

This repository is made by the [Euclid team](https://euclid.ee.duth.gr/), Democritus University of Thrace, Dept. of Electrical & Computer Engineering.

## Table Of Contents
* [Exploratory Data Analysis](notebooks/01.EDA.ipynb): Summarize the main characteristics of a dataset.
* [Feature Engineering](notebooks/02.Feature_Engineering_and_sklearn_baseline.ipynb): Select, transform and generate new features on a dataset.
* [Standardization and Normalization](notebooks/03.Preprocessing_Standardization_Normalization.ipynb): Scale variables. 
* [Imputation](notebooks/04.Handling_NaN_Imputation.ipynb): Handle NaN/null values.
* [Feature Selection](notebooks/05.Feature_Selection.ipynb): Reduce the number of input variables.
* [UnderSampling and OverSampling](notebooks/06.Undersampling_Oversampling.ipynb): Handle imbalanced datasets.
* [Deep Learning with PyTorch](notebooks/07.torch_training.ipynb): Build deep neural networks with PyTorch.
* [Cross Validation](notebooks/08.cross_validation.ipynb): Evaluate the generalization of a machine learning model.
* [Tuning with Grid-search](notebooks/09.Tuning_part1_sklearn.ipynb): Find optimal parameters on a given model.
* [Tuning with Bayesian Optimization](notebooks/09.Tuning_part2_torch.ipynb): Find optimal parameters on a given model using Bayesian optimization.
* [Introduction to Regression](notebooks/10.Houses_Regression.ipynb): Build regressor models.
* [Introduction to Community Detection on Graphs](notebooks/11.Clustering_KarateClub.ipynb): Find cluster in graph-based representations.
* [Dimensionality Reduction](notebooks/12.Dimensionality_reduction.ipynb): Project or transform data into a low-dimensional space.
* [Introducing CryptoNets](notebooks/13.Encrypted_Inference.ipynb): Evaluate a model on encrypted data.
* [Introducing MIA](notebooks/14.Membership_Inference_Attack.ipynb): Identify the training set that used to generate a predictive model.

# Installation
## Environment
We recommend the configuration using [Conda](https://docs.conda.io/en/latest/miniconda.html) and Python 3.8+ 
or using an external jupyter environment such as Colab or Kaggle.

## Dependencies
* imbalanced_learn
* matplotlib
* numpy
* pandas
* scikit_learn
* torch
* optuna
* seaborn
* tenseal
* networkx
* python-louvain
* notebook

## Installation with the specified requirements file.
```
pip install -r requirements.txt
```
