import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model.h5')

def predict(image):
    # Pre-process the image for the model
    image = cv2.resize(image, (48,48))
    image = image.reshape(1, 48, 48, 1)
    image = image / 255.0
    
    # Make predictions
    pred = model.predict(image)
    return pred

def main():
    st.title("Facial Expression Recognition")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Open a video stream
    video_capture = cv2.VideoCapture(0)
    
    while True:
        # Get the current frame from the video stream
        _, frame = video_capture.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face from the image
            face_cropped = gray[y:y+h, x:x+w]
            
            # Predict the facial expression
            pred = predict(face_cropped)
            pred = np.argmax(pred)
            label = emotions[pred]
            
            # Display the facial expression on the screen
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert the frame to a PIL image
        frame = Image.fromarray(frame)
        
        # Show the frame in the Streamlit app
        st.image(frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break