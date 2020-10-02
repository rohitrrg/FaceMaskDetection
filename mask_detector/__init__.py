from flask import Flask

app = Flask(__name__)

from mask_detector import routes
