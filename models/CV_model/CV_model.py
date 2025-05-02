import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('D:\AI Projects\models\CV_model\cup_pen_classifier.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the window name
window_name = 'Pen or Cup Classifier'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Define class labels
class_labels = ['Pen', 'Cup']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    img = cv2.resize(frame, (32, 32))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make prediction
    prediction = model.predict(img)[0][0]
    class_index = int(prediction > 0.5)
    class_label = class_labels[class_index]
    confidence = prediction if class_index == 1 else 1 - prediction

    # Prepare text to display
    text = f"{class_label}: {confidence:.2f}"
    
    # Display the result on the frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow(window_name, frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()