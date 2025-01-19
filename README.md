# Sign-Language-detection using mediapipe framework for enhanced performance
## 1. Command-Line Argument Parsing (get_args)

This function allows you to run the program with custom configurations using command-line arguments.

Arguments:

1. --device: Specifies the camera device (default: 0).
2. --width and --height: Set the camera frame resolution (default: 960x540).
3. --use_static_image_mode: Activates static image mode for MediaPipe.
4. --min_detection_confidence: Threshold for detecting hands (default: 0.7).
5. --min_tracking_confidence: Threshold for tracking hands (default: 0.5).
## 2. Main Workflow (main)

Key Components:

* Initialize Camera:

Configures the webcam resolution and prepares to capture video.
cv2.VideoCapture(cap_device): Opens the camera.

* MediaPipe Hands Model:

Initializes the hand detection model.
min_detection_confidence and min_tracking_confidence define thresholds for accurate hand detection and tracking.
* KeyPoint & PointHistory Classifiers:

These are custom-trained models that classify hand gestures and point histories.
Labels for these classifiers are read from CSV files (keypoint_classifier_label.csv and point_history_classifier_label.csv).
* FPS Calculation:

* CvFpsCalc: A custom utility to measure and smooth FPS (frames per second).
* Gesture History:

Uses deque (a double-ended queue) to store the history of finger gestures and point movements.
* Main Loop (Video Frame Processing):

Capture Frame: Reads video frames from the camera.
* Preprocessing:

Converts the frame to RGB (required for MediaPipe).

Detects hand landmarks using MediaPipe's Hands model.
* Bounding Box & Landmarks:
Bounding rectangles and landmark coordinates are calculated for detected hands.
* Classification:
Hand gestures are classified using the KeyPointClassifier.

Point histories are classified using the PointHistoryClassifier.

* Drawing:
Visualizes the bounding box, landmarks, and gesture information on the video frame.
* Display:

Shows the processed video in a window (cv2.imshow).
* Exit Condition:

Press ESC or q to exit.
## 3. Helper Functions
* calc_bounding_rect:
Calculates a bounding rectangle around the detected hand using the landmarks.

* calc_landmark_list:
Converts hand landmarks (normalized coordinates from MediaPipe) into pixel-based coordinates.

* pre_process_landmark:
Normalizes landmark coordinates relative to the first landmark.
Flattens the 2D coordinates into a 1D list and normalizes them.
* pre_process_point_history:
Converts the history of finger points into relative, normalized coordinates.
Flattens the 2D point history into a 1D list.
* logging_csv:
Logs hand landmarks and point histories into CSV files for dataset collection (training purposes).
Operates in three modes:
% mode 0: Does nothing.
% mode 1: Logs landmarks into keypoint.csv when a gesture number (0-9) is detected.
% mode 2: Logs point histories into point_history.csv.
## 4. Gesture Classification & Visualization
* Hand Gesture Classification:

Uses KeyPointClassifier to recognize specific hand gestures.
Gesture results are overlaid on the video frame.
* Point History Classification:

Classifies movement patterns (e.g., finger movement history).
Drawing Functions:

draw_bounding_rect, draw_landmarks, and others visualize bounding boxes, landmarks, and gesture labels on the screen.
## 5. Exit Mechanism
The program continuously processes frames in a loop.

Press ESC or q to break the loop, release the camera, and close the OpenCV window.

# Keypoint classifier custom model:

 Implementation of a key point classifier using a neural network model in TensorFlow and TensorFlow Lite (TFLite).

## 1. Dataset Preparation
The dataset contains features and labels:

* X_dataset: This is a matrix containing the input features extracted from the dataset (using 42 columns, 21 * 2).
* y_dataset: This is the vector containing the class labels (the target output for classification).
* train_test_split: The dataset is split into 75% training and 25% testing using the train_test_split function from sklearn.
  
## 2. Model Architecture
The model is a fully connected neural network designed to classify inputs into one of NUM_CLASSES (4 classes in this case).

* Input layer:
  
 tf.keras.layers.Input((21 * 2,)) specifies that the input has 42 features 
 
* Dropout layers:

Dropout is applied to prevent overfitting:

Dropout(0.2) drops 20% of nodes.

Dropout(0.4) drops 40% of nodes.

* Dense layers:
Dense(20, activation='relu'): Fully connected layer with 20 neurons and ReLU activation.
Dense(10, activation='relu'): Fully connected layer with 10 neurons and ReLU activation.
Dense(NUM_CLASSES, activation='softmax'): Output layer with 4 neurons (one per class) and softmax activation for probability distribution across classes.
## 3. TensorFlow Lite Interpreter
After training, the model is converted into TensorFlow Lite format (tflite_save_path) for optimized inference on edge devices.

## Steps:

* Model Loading:

Load the TFLite model using tf.lite.Interpreter(model_path=tflite_save_path).
* Allocate Tensors:

Prepare the model's tensors using interpreter.allocate_tensors().
* Get Input and Output Details:

Retrieve input details (e.g., tensor shape and data type) using input_details.
Retrieve output details using output_details.
* Input Data:

Set the input tensor using interpreter.set_tensor() with the first test sample (X_test[0]).
* Inference:

Invoke the interpreter using interpreter.invoke() to perform inference on the input.
## Output:

Retrieve the model's prediction using interpreter.get_tensor().
## 4. CPU Time Measurement
* The %%time magic command measures the execution time for the inference:

* CPU times: Measures user and system time.
* Wall time: Total elapsed time.
## 5. Prediction
 np.squeeze(tflite_results): Returns the predicted probabilities for all classes.
 
 np.argmax(np.squeeze(tflite_results)): Retrieves the index of the highest probability, indicating the predicted class.

## Summary
This script captures video input, detects hands using MediaPipe, classifies gestures using custom-trained models, and visualizes the results in real-time. It includes tools for collecting and logging data to enhance the training process.
