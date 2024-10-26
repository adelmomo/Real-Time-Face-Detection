# Face Detection

This repository demonstrates how to use the **YuNet** face detector model, a state-of-the-art convolutional neural network designed to detect faces at multiple scales in an image scene. The model is sourced from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), a collection of pre-trained models provided by OpenCV.

## Features

- **Face Detection Model**: The YuNet model weights and inference class are included for efficient face detection.

- **Utilities**: Helper functions for:
  - Drawing detected faces on an image or video frame.
  - Blurring detected faces for privacy.
  - Writing text onto the image.

## Face Detection Demo

The demo opens a webcam stream and runs the face detector in real time. You can choose to either draw the detected faces or blur them based on your needs.

- **Without Face Bluring**:

[Without Face Blurring Demo](https://github.com/adelmomo/Real-Time-Face-Detection/blob/main/face_detection_demo_without_blur.mp4)

- **With Face Bluring**:

[With Face Bluring Demo](https://github.com/adelmomo/Real-Time-Face-Detection/blob/main/face_detection_demo_with_blur.mp4)

## Prerequisites

Ensure that you have the following installed:
- Python 3.x
- OpenCV 4.10.0
- NumPy 2.0.1

You can install dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

**Running the Face Detector with Face Blurring**

To run the face detector and blur the faces detected in the scene, use the following command:

```bash
python demo.py -i 0 --face_blurring
```

**Running the Face Detector with Face Drawing**

To run the face detector and draw bounding boxes around detected faces, use the following command:

```bash
python demo.py -i 0
```

## Reference

This project utilizes the YuNet face detector model from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) repository.