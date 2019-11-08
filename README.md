# CS682
UMass Amherst CS682 Neural Networks Final Project

# Project
Can we predict the percieved entity from brain activity patterns for the passive viewing task?

### Data
* https://openneuro.org/datasets/ds000157/versions/00001
* https://github.com/OpenNeuroDatasets/ds000157

```
mkdir data
aws s3 sync --no-sign-request s3://openneuro.org/ds000157 data/ds000157/
```
(beats the alternatives, and the complicated datalad thing)

# Steps

# TODO
add requirements.txt
add docker?
add dvc? (not a bad idea potentially, beats the aws usage)

```
pip install pybids
pip install torch
pip install medicaltorch
pip install torchvision
```
