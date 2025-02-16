import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array


model = load_model(r'M:\Documents\VS CODE\image emo\final_emotion_detection_model.keras')
emotion_labels = ['Angry', 'Nothing', 'Happy', 'Sad']  # Update according to your classes


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)


def preprocess_face(roi_gray):
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_CUBIC)
    roi = roi_gray.astype('float32') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

while True:
    # Capture video frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
       
        roi_gray = gray[y:y+h, x:x+w]
        roi = preprocess_face(roi_gray)

        
        prediction = model.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Emotion Detector', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()