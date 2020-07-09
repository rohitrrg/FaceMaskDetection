import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model_new.h5')


def recognize(img):
    img = cv2.resize(img, (50, 50))
    x = np.array(img).reshape(-1, 50, 50, 1)
    x = tf.keras.utils.normalize(x, axis=1)
    pred = classifier.predict(x)
    pred = np.argmax(pred, axis=1)
    sol = None
    if pred == 1:
        sol = 'No Mask'
    elif pred == 0:
        sol = 'Mask'

    return sol


frame = cv2.imread('test2.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for face in faces:
    x1, y1, width, height = face
    x2, y2 = x1 + width, y1 + height
    crop_image = gray[y1:y2, x1:x2]
    res = recognize(crop_image)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, str(res), (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


cv2.imshow('face detection- Haarcascade', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
