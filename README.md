# Nexus – AI-Powered Waste Segregation (Version 2)

## Overview
Nexus is a computer vision and AI-based waste segregation system.  
It detects material type (plastic, metal, paper) using a trained ML model and sorts items using a robotic arm.

---

## Demo

![Demo](demo/screenshot.png)

---

## Problem Statement
Manual waste segregation is inefficient, labour-intensive, and hazardous.  
Nexus automates the classification and sorting of waste using machine learning and physical automation.

---

## Tech Stack
- Python  
- TensorFlow or Custom CNN  
- OpenCV  
- Robotic Arm with Servo Motors  
- Arduino Uno for inference

---

## Architecture  

![Architecture](docs/nexus_architecture.png)

---

## Dataset
- Over 1500 images  
- Classes: plastic, metal, paper  
- Preprocessing includes resizing, normalization, and augmentation  
- Train/Validation split: 80/20

---

## Model Details
- MobileNetV2 or Custom CNN  
- Adam optimizer  
- Categorical crossentropy loss  
- 20–35 training epochs  
- Batch size: 32

---

## Results

| Model         | Accuracy | Precision | Recall |
|---------------|----------|-----------|--------|
| MobileNetV2   | 92.4%    | 91.7%     | 92.1% |
| Custom CNN    | 89.3%    | 88.4%     | 89.0% |


---

## Limitations
- Accuracy reduces in poor lighting  
- Limited to three waste categories (Plastic,Metal,Paper)
- Robotic arm precision affected by irregular objects
- Bias in databse exists  

---

## Future Work
- Add more material classes  
- Use transformer-based models  
- Connect with IoT dashboards  
- Improve robotic positioning accuracy
- Use API based classification to reduce dependancy on predefined database

