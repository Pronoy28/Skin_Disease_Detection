from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

## Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.config["DEBUG"] = False

# Load your trained model

model = tf.keras.models.load_model('model/dis.h5')
    
print('Model loaded. Start serving...')

def model_predict(img_path, model):

    # Preprocessing the image
    x = np.asarray(Image.open(img_path).resize((224,224)))
    x = x/255
    x = np.asarray(x.tolist())
    x = np.expand_dims(x, axis=0)
    

    preds = model.predict(x)

    return preds

dis={0:'Actinic keratoses',1:'Basal cell carcinoma',2:'Benign keratosis-like lesions',3:'Dermatofibroma',4:'Melanoma',5:'Melanocytic nevi',6:'Vascular lesions'}

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html',preds=None)


@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, './uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        #print(preds)
        # # Process your result for human
        
        preds = np.argmax(preds)
        preds=dis[preds]
        print(preds)
        #print(preds) 
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return render_template('index.html',preds=preds)
    return None




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
