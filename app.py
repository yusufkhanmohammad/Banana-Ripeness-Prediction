from __future__ import division, print_function
import sys
import os
from PIL import Image
import glob
import pickle
import re
from tensorflow.keras.models import load_model 

import numpy as np


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'banana.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary

def model_predict(img_path, model):
    
    # Preprocessing the image
    color_img = np.asarray(Image.open(img_path)) / 255
    img = np.mean(color_img, axis=2)
    preds = model.predict(img.reshape(1,32,32,1))

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1) 
        #            # Simple argmax
        
        a = preds[0][0]
        b = preds[0][1]
        c = preds[0][2]

        if (a >= b) and (a >= c):
            largest = a
        elif (b >= a) and (b >= c):
            largest = b
        else:
            largest = c
        

        if largest == a:
            result = "Raw"
        if largest == b:
            result = "Ripe"
        if largest == c:
            result = "Over Ripe"

        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

