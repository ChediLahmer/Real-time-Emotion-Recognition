# Emotion Recognition System

# **Documentation**

## **Part 1: Training and Model Generation**

### **Objective**

This section outlines the process of training a convolutional neural network (CNN) for emotion recognition. The pre-trained MobileNetV2 model is fine-tuned on a custom dataset containing images of faces labeled with seven different emotions.

### **Code Explanation**

```python
pythonCopy code
Datadirectory = "training/"
Classes = ["0", "1", "2", "3", "4", "5", "6"]

# ... (imports and variable declarations)

# Function to create training data
def create_training_Data():
    # ... (code to create balanced training dataset)

# Call the function to create training data
create_training_Data()

# Shuffle the training data
random.shuffle(training_Data)

# Prepare input (x) and output (Y) data
x = np.array(x).reshape(-1, img_size, img_size, 3)
x = x / 255.0
Y = np.array(y)

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output

# Modify the model architecture
final_output = layers.Dense(128)(base_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation="softmax")(final_output)

# Create a new model
new_model = keras.Model(inputs=base_input, outputs=final_output)

# Compile and train the new model
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
new_model.fit(x, Y, epochs=25)

# Save the trained model
new_model.save('my_mmodel_64p35.h5')

```

### **Usage**

1. Ensure the existence of a well-structured dataset in the **`training/`** directory.
2. Adjust the **`Datadirectory`**, **`Classes`**, and other parameters as needed.
3. Run the script to create and train the emotion recognition model.
4. The trained model will be saved as 'my_mmodel_64p35.h5' for future use.

## **Part 2: Real-Time Emotion Recognition**

### **Objective**

This section demonstrates the real-time application of the trained emotion recognition model on webcam input. The OpenCV library is utilized to capture video frames, detect faces, and predict emotions.

### **Code Explanation**

```python
pythonCopy code
# ... (imports and variable declarations)

# Load the pre-trained emotion recognition model
new_model = tf.keras.models.load_model('my_mmodel_64p35.h5')

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# ... (main loop for real-time emotion recognition)

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

```

### **Usage**

1. Ensure the trained model file ('my_mmodel_64p35.h5') is available.
2. Run the script to open the webcam and start real-time emotion recognition.
3. The script will display the video feed with emotion labels on detected faces.
4. Press 'q' to exit the application.

## **Note**

- The emotion classes are represented as follows: ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"].
- Ensure the proper functioning of the webcam and face detection cascade classifier.

## **References**

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MobileNetV2 Documentation](https://keras.io/api/applications/mobilenet/#mobilenetv2-function)