from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
sys.path.append(os.path.abspath('..'))

# ???

app = Flask(__name__)

global model, graph = keras.models.


def convert_image(img_data):
    img_str = re.search(r'base64,(.*)', img_data).group(1)
    with open('out.png', 'wb') as img:
        img.write(img_str.decode('base64'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img_data = request.get_data()
    convert_image(img_data)
    x = imread('out.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    with graph.as_default():
        out = model.predict(x)
        response = np.array_str(np.argmax(out, axis=1))
        return response
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
    