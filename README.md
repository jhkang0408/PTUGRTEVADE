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

> **Note:** If you simply place the *Target Labeled Data* into `Unity/Assets/Data/`, you can skip this step. This import process is intended for cases where you need to import the *raw motion capture data* — for example, to label targets yourself or to annotate additional features.

![Data Import](Media/DataImport.gif)

1\) Open the data scene (`Unity/Assets/Data/Scene.unity`).<br>
2\) From the top menu bar, open **Motion Data Processing → Motion Importer**.<br>
3\) In the **Motion Importer** window, select the split (**Train** / **Test**) and choose which data to import.<br>
4\) Click **Import Motion**.<br>
5\) Verify that the data has been correctly added under `Unity/Assets/Data/`.

## Data Visualization

| | |
|---|---|
| ![Explaining with pointing](Media/Explaining%20with%20pointing.gif) | ![Touching and gazing at two targets in sequence](Media/Touching%20and%20gazing%20at%20two%20targets%20in%20sequence.gif) |
| **Explaining with pointing** | **Touching and gazing at two targets in sequence** |
| ![Referencing multiple targets with diverse gestures (example 3)](Media/Referencing%20multiple%20targets%20with%20diverse%20gestures%20%28example3%29.gif) | ![Referencing multiple targets with diverse gestures (example 2)](Media/Referencing%20multiple%20targets%20with%20diverse%20gestures%20%28example2%29.gif) |
| **Referencing multiple targets with diverse gestures (example 3)** | **Referencing multiple targets with diverse gestures (example 2)** |
| ![Referencing multiple targets with diverse gestures (example 1)](Media/Referencing%20multiple%20targets%20with%20diverse%20gestures%20%28example1%29.gif) | ![Touching a single target with gaze shift](Media/Touching%20a%20single%20target%20with%20gaze%20shift.gif) |
| **Referencing multiple targets with diverse gestures (example 1)** | **Touching a single target with gaze shift** |

## Data Export

_Coming soon._

## Acknowledgements

This repository contains pieces of code from the following repositories:
- [AI4Animation: Deep Learning for Character Control](https://github.com/sebastianstarke/AI4Animation/tree/master)
