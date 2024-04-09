from flask import Flask, flash, request, redirect, url_for,render_template
from flask import send_file
import os
from flask_cors import CORS, cross_origin
from src.cnnClassifier.utils.common import decodeImage
from src.cnnClassifier.pipeline.hide_prediction import HidePredictionPipeline
from src.cnnClassifier.pipeline.reveal_prediction import RevealPredictionPipeline
from PIL import Image
import socket

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
        self.coverout = "static\images\Coverout.png"
        self.toreveal = r"static\images\toreveal.png"
        self.secretout = r"static\images\secretout.png"
        self.hideclassifier = HidePredictionPipeline(self.secret, self.cover)
        self.revealclassifier = RevealPredictionPipeline()

clApp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home(): # change to login.html
    return render_template('home.html', prediction=None, confidence=None, img_path=None)



@app.route("/hide") # replaced index with hide
@cross_origin()
def hide():
    return render_template('hide.html')

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
        cover = request.files['cover']  # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if secret.filename == '' or cover.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if (secret and allowed_file(secret.filename)) or (cover and allowed_file(cover.filename)):
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
            toreveal.save(clApp.toreveal)
            secretout = clApp.revealclassifier.predict(clApp.toreveal)
            Image.fromarray(secretout).save(clApp.secretout)

    return render_template("revealed.html") # revealed.html page is still not done


@app.route("/download_image")
@cross_origin()

def download_image():
    # Define the path to your image file
    image_path = "static/images/Coverout.png"
    # Send the file for download
    return send_file(image_path, as_attachment=True)
if __name__ == "__main__":
    #db.init_app(app) # initialize db after app creation
    app.run(host='0.0.0.0', port=80)  # for Azure




#for web socket we will deal with it later
"""import smtplib # lib for notif sending

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email
from flask_sqlalchemy import SQLAlchemy
import bcrypt  # For secure password hashing"""

"""#begin login auth here with postgresql
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@host:port/database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking
app.secret_key = 'goodnight'  # Replace with a strong secret
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

# Login form with CSRF protection
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            # User authenticated, create session (replace with more secure session management)
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid login credentials', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.before_first_request
def create_tables():
    db.create_all()
@app.route('/')
@app.route('/home')
@app.route('/hide')
@app.route('/reveal')
def protected_routes():
    if 'user_id' not in session:
        return redirect(url_for('login'))"""

#################################################################


"""HOST = '0.0.0.0'
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        #send image
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(image)"""