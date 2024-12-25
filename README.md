<p style="display: inline">
  <img alt="Static Badge" src="https://img.shields.io/badge/Python-3.10.9-yellow?logo=python">
  <img alt="Static Badge" src="https://img.shields.io/badge/Pytorch-1.12.1-blue?logo=pytorch">
  <img alt="Static Badge" src="https://img.shields.io/badge/Numpy-1.24.3-red?logo=numpy">
</p>

# Direct
This repository contains codes of Direct approaches in [Multi-Task Learning Approaches for Music Similarity Representation Learning Based on Individual Instrument Sounds](http://www.apsipa2024.org/files/papers/333.pdf).

## About Branches
The main branch contains the core codes of Direct approaches. The code for each specific method is in the branch.
The following branch contains the codes of the method that is proposed in [Multi-Task Learning Approaches for Music Similarity Representation Learning Based on Individual Instrument Sounds](http://www.apsipa2024.org/files/papers/333.pdf).
- Direct-Reconst

## About files
Explain about files in the project

- configs
  - The files in this folder set hyperparapeters of the model.
- dataset
  - The files in this folder are written the codes for preparing datasets.
- model
  - The files in this folder contains model structure codes (class nn.Module) and training codes (class LightningModule)
- source
  - The files in this folder contains excutable files for train and eval.
- utils
  - The files in this folder contains additional setting codes of the module. Especially func.py contains miscellaneous function codes.

