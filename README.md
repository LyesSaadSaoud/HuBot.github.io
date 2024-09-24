# HuBot

This repository contains the source code for the [Lyes Saad Saoud](https://github.com/LyesSaadSaoud/HuBot.github.io).

# HuBot: A Biomimicking Mobile Robot for Non-Disruptive Bird Behavior Study and Ecological Conservation
# Abstract
The Houbara bustard, a critically endangered avian species, poses significant challenges for researchers due to its elusive nature and sensitivity to human disturbance. Traditional research methods, often reliant on human observation, have yielded limited data and can inadvertently impact the bird’s behavior. To overcome these limitations, we propose for the first time HuBot, a biomimetic mobile robot designed to seamlessly integrate into the Houbara's natural habitat. Equipped with advanced real-time deep learning algorithms, including YOLOv9 for detection, MobileSAM for segmentation, and ViT for depth estimation, HuBot autonomously tracks individual birds, providing unprecedented insights into their movement patterns, social interactions, and habitat use. By accurately detecting and localizing the houbara, HuBot contributes to a deeper understanding of the Houbara's ecology and informs critical conservation decisions. The robot's biomimetic design, including life-like appearance and movement, minimizes disturbance, allowing for long-term, continuous monitoring without compromising data quality. Rigorous testing, including extensive laboratory experiments and captive trials with real Houbara birds, validated HuBot's performance and its potential to revolutionize the study of this enigmatic species. Through the deployment of HuBot, we aim to provide essential information for developing effective conservation strategies to safeguard the future of the Houbara bustard.

# **HuBot: A Biomimicking Mobile Robot for Non-Disruptive Bird Behavior Study and Ecological Conservation**

## **Experimental Setup and Model Implementations**

To achieve seamless integration and reliable performance of HuBot in the field, we employed advanced segmentation and detection models including Detectron2, YOLOv5-seg, YOLOv8-seg, and MobileSAM from Ultralytics. This section provides a comprehensive guide on the setup, configuration, and deployment of these models, ensuring replicability of the experiments conducted.

### **1. Detectron2 Setup and Usage**

**Detectron2** is an open-source library developed by Facebook AI Research (FAIR) for state-of-the-art object detection and segmentation. The following steps outline the installation and configuration of Detectron2 used in our study:

#### **Installation Steps:**

1. **Install Compatible PyTorch:**
    ```bash
    # Replace '11.7' with your CUDA version or use 'cpu' if you don't have CUDA
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    ```

2. **Install Detectron2:**
    ```bash
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

3. **Verify Installation:**
    ```bash
    python -c "import detectron2; print(detectron2.__version__)"
    ```

#### **Running Inference with Detectron2:**

The following Python script was used for running inference with a pre-trained Mask R-CNN model:

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import cv2
import numpy as np
from pathlib import Path
import time

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
# Code continues with inference and mask saving...
2. YOLOv5 Instance Segmentation
YOLOv5-seg was utilized for instance segmentation to detect and track Houbara birds effectively. Below are the setup and execution steps:

Setup Steps:
Clone the Ultralytics Repository:


git clone https://github.com/ultralytics/yolov5.git
cd yolov5
Install Requirements:


pip install -r requirements.txt
YOLOv5 Directory Structure:
The segmentation functionality resides within the segment directory:

kotlin
Copy code
├── classify
├── data
│   ├── hyps
│   └── xView.yaml
├── models
│   ├── hub
│   └── yolov5x.yaml
├── runs
│   ├── detect
│   └── predict-seg
├── segment
│   ├── predict.py
│   ├── train.py
│   └── val.py
├── utils
└── benchmarks.py
Running YOLOv5 Instance Segmentation:

python segment/predict.py --weights yolov5n-seg.pt --source ../input/images --view-img
3. YOLOv8 Instance Segmentation
YOLOv8 enhances detection and segmentation with more refined algorithms and faster processing. The following outlines its installation and usage:

Setup and Run YOLOv8-seg:
Install YOLOv8:


pip install ultralytics==8.2.87
Run YOLOv8 Segmentation:

python segment/predict.py --weights yolov8n-seg.pt --source ../input/images
4. YOLOv9 and MobileSAM (Ultralytics)
YOLOv9 and MobileSAM models were implemented to provide a mobile-optimized approach to segmentation, ideal for field deployment scenarios:

Installation:
Clone and Install YOLOv9 Repository:


git clone https://github.com/ultralytics/yolov9.git
cd yolov9
pip install -r requirements.txt
Follow Ultralytics Guide for MobileSAM:

Refer to the official Ultralytics MobileSAM Guide.

5. Performance Evaluation and Inference Timing
We evaluated the inference time per image for each model to ensure real-time capabilities. The average inference times were recorded, demonstrating the efficiency of the algorithms deployed on HuBot.

Sample Command for YOLOv5:


python segment/predict.py --weights yolov5n-seg.pt --source ../input/images --view-img
Sample Command for YOLOv8:


yolo segment predict model=yolov8n-seg.pt source="../input/images"
This documentation ensures the reproducibility of the experimental setups and facilitates further research and development in non-disruptive ecological monitoring using biomimetic robotics.

If you find Hubot useful for your work please cite:
```
@article{park2021nerfies
  author    = {Lyes},
  title     = {Hubot},
  journal   = {xxx},
  year      = {2024},
}
```


