
#import flask library
from flask import Flask, render_template, request

from scipy.misc import imsave, imread, imresize
import numpy as np 
import keras.models
import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
from gevent.wsgi import WSGIServer
from keras.preprocessing import image
from werkzeug.utils import secure_filename
#from load import *

#Initialize the app from Flask
app = Flask(__name__)

model = keras.models.load_model('models/mnist_natacha_model.h5')
print(model)

c = [1,2,3]

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'POST':
        
    
        f = request.files['file']
       
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
        
        
        f.save(file_path)
        
        img = image.load_img(file_path, target_size=(28, 28), grayscale=True)
        
        x = image.img_to_array(img)
        x = np.true_divide(x, 255)
        x = x.reshape(28, 28, 1)
        x = np.expand_dims(x, axis=0)
        result = model.predict(x)
        result = np.argmax(result)
        
        return render_template('predict.html', result=result)
    return render_template('predict.html') 


if __name__ == "__main__":
    #app.run(port=8085, debug=True)
    http_server = WSGIServer(('', 8085), app)
    http_server.serve_forever()

    

    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8085))
    app.run(host='0.0.0.0', port=port)
    
    


