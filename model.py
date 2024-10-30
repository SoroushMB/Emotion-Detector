import numpy as np
import matplotlib.pyplot as plt
import argparse, cv2, os
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command-Line Argument
ArgParses = argparse.ArgumentParser()
ArgParses.add_argument('--mode', help='train/display')
Mode = ArgParses.parse_args().mode

# Plots accuracy and loss curves (function commented out in original code)
# def PlotModelHistory(model_history):
#     # Code for plotting model history

# Define data generators
TrainDirectory = 'data/train'
ValDirectory = 'data/test'

NumberTrain = 28709
NumberValidate = 7178
BatchSize = 64
NumberEpoch = 50

TrainDataGen = ImageDataGenerator(rescale=1./255)
ValDataGen = ImageDataGenerator(rescale=1./255)

TrainGenerator = TrainDataGen.flow_from_directory(
    TrainDirectory,
    target_size=(48,48),
    batch_size=BatchSize,
    color_mode="grayscale",
    class_mode='categorical')

ValidationGenerator = ValDataGen.flow_from_directory(
    ValDirectory,
    target_size=(48,48),
    batch_size=BatchSize,
    color_mode="grayscale",
    class_mode='categorical')

# Build the model
Model = Sequential()

Model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48,48,1)))
Model.add(Conv2D(64, (3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Conv2D(128, (3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Conv2D(128, (3, 3), activation='relu'))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

Model.add(Flatten())
Model.add(Dense(1024, activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(7, activation='softmax'))

Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
if Mode == "train":
    Model.fit_generator(
        TrainGenerator,
        steps_per_epoch=NumberTrain // BatchSize,
        epochs=NumberEpoch,
        validation_data=ValidationGenerator,
        validation_steps=NumberValidate // BatchSize)
    Model.save_weights('model.h5')

# Emotion detection code
def detect_emotion(frame, model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        maxindex = int(np.argmax(prediction))
        return emotion_dict[maxindex]

# Initialize a deque to store the last N predictions
N = 10  # Window size for moving average
emotion_history = deque(maxlen=N)

# Function to get the most common emotion in the history
def get_dominant_emotion(emotion_history):
    if len(emotion_history) == 0:
        return None
    return max(set(emotion_history), key=emotion_history.count)

# Capture video from the camera
if Mode == "display":
    Model.load_weights('model.h5')
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion
        emotion_label = detect_emotion(frame, Model)

        if emotion_label is not None:
            # Add the current emotion to the history
            emotion_history.append(emotion_label)

            # Get the dominant emotion
            dominant_emotion = get_dominant_emotion(emotion_history)

            # Display the dominant emotion on the frame
            if dominant_emotion:
                cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
