# Emotion Detection Using CNN and OpenCV

This project implements a real-time emotion detection system using a Convolutional Neural Network (CNN) and OpenCV. The system classifies emotions from facial expressions captured via webcam and displays the dominant emotion on the screen. The model is trained on the FER-2013 dataset and supports the following emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features
- **Real-Time Emotion Detection**: Captures live video feed and predicts facial emotions in real-time.
- **Deep CNN Model**: Uses a Convolutional Neural Network architecture with batch normalization and dropout for stable and accurate predictions.
- **Temporal Smoothing**: Averages recent predictions to reduce fluctuations and provide stable emotion output.
- **Data Augmentation**: Enhances the training dataset with rotations, shifts, and flips for better model generalization.
- **Learning Rate Scheduling**: Adjusts learning rate dynamically during training for optimized performance.
- **Class Balancing**: Handles imbalanced emotion classes by applying weighted loss for underrepresented classes.

## Dataset
The **FER-2013** dataset is used to train the CNN model. This dataset includes grayscale images of faces with seven emotion labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

- **Download FER-2013**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## Requirements
- Python 3.x
- OpenCV
- TensorFlow
- Keras
- NumPy

Install the dependencies using:
```bash
pip install opencv-python tensorflow keras numpy
