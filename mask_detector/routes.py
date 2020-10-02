from flask import render_template, Response, url_for
from mask_detector import app
from mask_detector.detector import MaskDetector
from mask_detector.detector import MaskDetector


@app.route('/')
def home():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.detect()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(MaskDetector()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')