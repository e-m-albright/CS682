# CS682
UMass Amherst CS682 Neural Networks Final Project

# Project
Can we predict the percieved entity from brain activity patterns for the passive viewing task?

### Data

##### Citation
Smeets PA, Kroese FM, Evers C, de Ridder DT.
Behav Brain Res. 2013 Jul 1;248:41-5. doi: 10.1016/j.bbr.2013.03.041. Epub 2013 Apr 8.
http://www.ncbi.nlm.nih.gov/pubmed/23578759

Full paper
https://pdfs.semanticscholar.org/e18e/fa2271db963956c35ce1cb8e5b58737dd6f1.pdf
##### Sources
* (openneuro) https://openneuro.org/datasets/ds000157/versions/00001
* (datalad) https://github.com/OpenNeuroDatasets/ds000157

```
# To download from openneuro
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
pip install nibabel
pip install nilearn
pip install torch
pip install medicaltorch
pip install torchvision
```
