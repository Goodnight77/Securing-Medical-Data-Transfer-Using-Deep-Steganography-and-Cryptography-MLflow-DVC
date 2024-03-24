from flask import Flask, flash, request, redirect, url_for,render_template

import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.hide_prediction import HidePredictionPipeline
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.secret = "secret.jpg"
        self.cover = "cover.jpg"
        self.coverout = "Coverout.jpg"
        self.hideclassifier = HidePredictionPipeline(self.secret, self.cover)

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html', prediction=None, confidence=None, img_path=None)

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    # os.system("dvc repro")
    return "Training done successfully!"

@app.route("/hidepredict", methods=['POST'])
@cross_origin()
def hideprediction():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'secret' not in request.files or 'cover' not in request.files:
            flash('No file part')
            return redirect(request.url)
        secret = request.files['secret']
        cover = request.files['cover']        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if secret.filename == '' or cover.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if (secret and allowed_file(secret.filename))or (cover and allowed_file(cover.filename)):
            secret.save( os.path.join("static\images",clApp.secret))
            cover.save(os.path.join("static\images",clApp.cover))
            coverout=clApp.hideclassifier.predict()
            Image.fromarray(coverout).save(os.path.join("static\images",clApp.coverout))
    
    
    return render_template("result.html", predictions=str(coverout))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)  # for Azure
