import cv2
import numpy as np
import keras

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model('model_new.h5')


class MaskDetector(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def detect(self):

        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for face in faces:
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height

            crop_image = gray[y1:y2, x1:x2]

            img = cv2.resize(crop_image, (50, 50))
            x = np.array(img).reshape(-1, 50, 50, 1)
            x = keras.utils.normalize(x, axis=1)

            pred = model.predict(x)
            pred = np.argmax(pred, axis=1)

            sol = None
            if pred == 1:
                sol = 'Mask'
            elif pred == 0:
                sol = 'No Mask'

            if sol == 'Mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif sol == 'No Mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(frame, str(sol), (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
