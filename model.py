import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model from file
model = load_model('model50.keras')
print("Model loaded successfully!")

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Load Haar cascade for face detection
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Extract face region and preprocess
        roi_gray = gray[y:y+h, x:x+w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
        cropped_img = cropped_img / 255.0  # normalize like in training

        # Predict emotion
        prediction = model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        confidence = prediction[0][maxindex]

        # Show predicted emotion and confidence
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show frame with annotations
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Emotion detection stopped.")
