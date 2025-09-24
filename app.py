from flask import Flask, render_template, request, redirect, url_for,session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import Sequential
from mantranet_model import load_trained_model
import numpy as np
import os
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from numpy import loadtxt
from PIL import Image, ImageChops, ImageEnhance
import itertools
from sqlalchemy.orm import backref
from io import BytesIO
import re
import cv2

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret-key'
#app.run(debug=True)
# The user must download ManTraNet_Ptrain4.h5 and place it in the root directory.
# Link: https://raw.githubusercontent.com/neelanjan00/Image-Forgery-Detection/master/ManTraNet_Ptrain4.h5
model = load_trained_model()
# app.config['UPLOAD_FOLDER'] = r'uploads'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    images = db.relationship('Images', backref='user', lazy=True)
    # def __repr__(self):
    #     return f'<User {self.username}>'

class Images(db.Model):
    image_id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    filename = db.Column(db.String(50), nullable=False)
    data = db.Column(db.LargeBinary)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # def __repr__(self):
    #     return f'<Image {self.filename}>'



@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/login')
# def login():
#     return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'confirm_password' in request.form:
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        account = User.query.filter_by(username=username).first()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must not contain any special characters!'
        elif not username or not password or not confirm_password:
            msg = 'Please fill out the form !'
        elif password != confirm_password:
            msg = 'Passwords do not match.'
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            msg = 'You have successfully registered!'
            # return render_template('signup.html', msg=msg)
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
        return redirect(url_for('login'))
    return render_template('signup.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        session['logged_in'] = True
        session['username'] = username
        global user
        user = User.query.filter_by(username=username).first()
        if not user or user.password != password:
            error = 'Invalid username or password.'
            return render_template('login.html', error=error)
        #return redirect(url_for('upload'))
        return render_template('home.html')
    return render_template('login.html')


@app.route('/upload', methods=['GET','POST'])
def upload_image():
    msg = ''
    # Check if a file was submitted
    if 'file' not in request.files:
        msg = 'No file submitted'
        return render_template('home.html',user=user,msg=msg)

    # Get the submitted file
    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        msg = 'Empty file submitted'
        return render_template('home.html',user=user,msg=msg)
    file.save('static/' + file.filename)

    # Preprocess the image using ManTraNet's approach
    image_array = prepare_image(file)

    # Predict the mask
    predicted_mask = model.predict(image_array)[0, ..., 0]

    # Save the original image and the predicted mask overlay
    file.seek(0) # Reset file pointer after reading in prepare_image
    original_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # To display, we might need to use matplotlib to save the overlay
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(original_image)
    ax.imshow(predicted_mask, cmap='jet', alpha=0.5)

    mask_filename = 'mask_' + file.filename
    mask_path = os.path.join('static', mask_filename)
    fig.savefig(mask_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # For now, just return the path to the mask
    return render_template('home.html', result_mask=mask_filename, images=file.filename)

def prepare_image(image_file):
    # Read image using OpenCV
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    # ManTraNet preprocessing
    # Convert to float32, normalize to [-1, 1], and add batch dimension
    x = np.expand_dims(img.astype('float32') / 255. * 2 - 1, axis=0)

    # Reset the file pointer to be able to read it again for saving
    image_file.seek(0)

    return x
@app.route('/view', methods=['GET','POST'])
def show_image():
    user = User.query.filter_by(username=session['username']).first()
    image = Images.query.filter_by(user_id=user.id)
    return render_template('view_image.html', images=image)

@app.route('/about', methods=['GET','POST'])
def about_us():
    return render_template('aboutus.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('index'))    

if __name__ == '__main__':
    # This block is intentionally left empty.
    # The app is run via a WSGI server like gunicorn,
    # and the database is initialized via a separate command.
    pass
