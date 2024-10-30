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
```

## Project Structure
```
├── data
│   ├── train          # Contains training images organized in subfolders by emotion class
│   └── test           # Contains test images organized in subfolders by emotion class
├── Emotions.py        # Main Python file with model training and real-time emotion detection
├── README.md          # Project documentation
```

## Usage
1. Download and Organize Dataset: Place the FER-2013 dataset images in the data/train and data/test directories, organized into subfolders by emotion label.

2. Train the Model: Run the following command to train the model with data augmentation, batch normalization, and class weights:

```bash
python Emotions.py
```
The model will be saved as `emotion_model.h5` after training.

3. Run Real-Time Emotion Detection: After training, the model will automatically start real-time emotion detection using the webcam:
```bash
python Emotions.py
```

4. Exit: Press `q` to quit the real-time detection window.

## Code Overview

- **Data Augmentation:** The code uses Keras' ImageDataGenerator to augment images for more robust training.

- **CNN Model:** The CNN architecture includes three convolutional layers with batch normalization and dropout for improved accuracy and stability.

- **Learning Rate Scheduler:** The ReduceLROnPlateau callback dynamically adjusts the learning rate based on validation loss.

- **Temporal Smoothing:** A deque stores the most recent predictions, and the most common emotion is displayed for a smoother output.

- **Class Weights:** Adjusts the loss function to account for class imbalance.

## Example Output

The system detects the user's face, predicts their emotion, and displays it on the live video feed. The output updates smoothly thanks to temporal smoothing, avoiding rapid emotion changes.

## Future Improvements

- **Advanced Pre-trained Models**: Experiment with transfer learning using models like ResNet or MobileNet.
- **Additional Emotion Classes**: Include more nuanced emotions by using datasets like AffectNet.
- **Fine-tuning for Real-World Use**: Apply the model to diverse datasets for improved accuracy in real-world conditions.
