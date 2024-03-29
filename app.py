from flask import Flask, flash, request, redirect, url_for,render_template
from flask import send_file

import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.hide_prediction import HidePredictionPipeline
from cnnClassifier.pipeline.reveal_prediction import RevealPredictionPipeline
from PIL import Image
import numpy as np
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
        self.secret = "static\images\secret.jpg"
        self.cover = "static\images\cover.jpg"
        self.coverout = "static\images\Coverout.jpg"
        self.toreveal = r"static\images\toreveal.jpg"
        self.secretout = r"static\images\secretout.jpg"
        self.hideclassifier = HidePredictionPipeline(self.secret, self.cover)
        self.revealclassifier = RevealPredictionPipeline()

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('home.html', prediction=None, confidence=None, img_path=None)

@app.route("/hide")
@cross_origin()
def hide():
    return render_template('index.html')

@app.route("/reveal")
@cross_origin()
def reveal():
    return render_template('reveal.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

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
            secret.save( clApp.secret)
            cover.save(clApp.cover)
            coverout=clApp.hideclassifier.predict()
            Image.fromarray(coverout).save(clApp.coverout)
    
    
    return render_template("result.html", predictions=str(coverout))

@app.route("/revealpredict", methods=['POST'])
@cross_origin()
def revealprediction():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'toreveal' not in request.files :
            flash('No file part')
            return redirect(request.url)
        toreveal = request.files['toreveal']
        # empty file without a filename.
        if toreveal.filename == '' :
            flash('No selected file')
            return redirect(request.url)
        if (toreveal and allowed_file(toreveal.filename)):
            toreveal.save( clApp.toreveal)
            secretout=clApp.revealclassifier.predict(clApp.toreveal)
            Image.fromarray(secretout).save(clApp.secretout)
    
    
    return render_template("revealed.html") # revealed.html page is still not done
@app.route("/download_image")
@cross_origin()

def download_image():
    # Define the path to your image file
    image_path = "static/images/Coverout.jpg"
    # Send the file for download
    return send_file(image_path, as_attachment=True)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)  # for Azure
