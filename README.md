# Part-wise Translation of Upper-body Gestures via Real-time Target Estimation for Virtual Avatars in Dissimilar Environments (Official Implementation)
![Python](https://img.shields.io/badge/Python->=3.8.12-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.11.0-Red?logo=pytorch)  ![Unity](https://img.shields.io/badge/Unity-2020.3.9f1-black?logo=unity&logoColor=white)

> **Part-wise Translation of Upper-body Gestures via Real-time Target Estimation for Virtual Avatars in Dissimilar Environments**<br>
> Jiho Kang, Jihun Shin, Kwang Bin Lee, Dong Hwan Shin, Yuna Choi, and [Sung-Hee Lee](https://scholar.google.com/citations?hl=en&user=AVII4wsAAAAJ)<br>
> IEEE TVCG, Major Revision

## Checklist

### Unity
- [x] Data import
- [x] Data processing (e.g., target labeling)
- [x] Data visualization
- [x] Data export (for training/evaluation)
- [ ] Real-time inference

### DeepLearning
- [x] Training
- [x] Model
- [x] Real-time inference
- [ ] Evaluation

## Data

Download the data from Google Drive and unzip each archive to the specified location:

| Data | Description | Location |
| --- | --- | --- |
| [Motion Capture Data](https://drive.google.com/file/d/1EzcetggbN8AMRDQDppRYPFkCk7DHdYQO/view?usp=sharing) | Raw motion capture data | `Unity/` |
| [Target Labeled Data](https://drive.google.com/file/d/1qqO5hcnfvh7bA3q22fnZ2oR4pdn2oCWR/view?usp=sharing) | Target-labeled motion capture data | `Unity/Assets/Data/` |

## Data Import

Before getting started, please refer to the GIF below.

![Data Import](Media/DataImport.gif)

1. Open the data scene (`Unity/Assets/Data/Scene.unity`).
2. From the top menu bar, open **Motion Data Processing → Motion Importer**.
3. In the **Motion Importer** window, select the split (**Train** / **Test**) and choose which data to import.
4. Click **Import Motion**.

## Data Export

_Coming soon._

## Acknowledgements

This repository contains pieces of code from the following repositories:
- [AI4Animation: Deep Learning for Character Control](https://github.com/sebastianstarke/AI4Animation/tree/master)
