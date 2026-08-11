# 🧶 Smart Yarn Quality Analysis

## Project Overview

Smart Yarn Quality Analysis is a computer vision based application designed to detect yarn defects using a YOLO object detection model.

The system analyzes yarn images and videos and identifies two types of yarn defects:

- Loop Fiber
- Protruding Fiber

## Objectives

- Automatically detect yarn defects.
- Reduce manual yarn inspection.
- Identify defect type.
- Display prediction confidence.
- Analyze both images and videos.
- Provide an easy-to-use web application.

## Technologies Used

- Python
- YOLO
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pillow
- GitHub

## Detected Defects

The trained model detects:

1. loop_fiber
2. protruding_fiber

## Features

### Image Analysis

Users can upload JPG, JPEG, or PNG yarn images.

The application displays:

- Detected defect
- Confidence score
- Detection bounding box
- Result image

### Video Analysis

Users can upload MP4, AVI, MOV, or MKV videos.

The application analyzes video frames and detects yarn defects.

## Dataset

- Training images: 1440
- Validation images: 138
- Test images: 66
- Number of classes: 2

## Model

The project uses a YOLO object detection model trained on a yarn defect dataset.

Classes:

- 0 → loop_fiber
- 1 → protruding_fiber

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
