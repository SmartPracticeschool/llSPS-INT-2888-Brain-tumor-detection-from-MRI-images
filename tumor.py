from __future__ import division, print_function
from cloudant import Cloudant
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from gevent.pywsgi import WSGIServer

import sys
import os
import glob
global model,graph
import tensorflow as tf
graph = tf.get_default_graph()
app = Flask(__name__)

#MODEL_PATH = 'cnn_Yes_No.h5'

#model = load_model('cnn_Yes_No.h5')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x1= np.expand_dims(x,axis=0)
    preds = model.predict_classes(x1)
    return preds


@app.route('/', methods=['GET'])
def index():
     return render_template('welcome.html')
 
@app.route('/start', methods=['GET'])
def start():
     return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model = load_model('cnn_Yes_No.h5')

        preds = model_predict(file_path, model)
        ls=["No","Yes"]
        result = ls[preds[0][0]]             
        return result
    return None


if __name__ == '__main__':
      port = int(os.getenv('PORT', 5000))
     #app.run(host='0.0.0.0', port=port, debug=True)
      http_server = WSGIServer(('0.0.0.0', port), app)
      http_server.serve_forever()
     #app.run(debug=True)