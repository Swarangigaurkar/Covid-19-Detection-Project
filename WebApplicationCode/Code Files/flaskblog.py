import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from load import *
import cv2
import tensorflow as tf
from keras.preprocessing import image

# UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

global graph, model

model, graph = init()

app = Flask(__name__)
app.secret_key = "hello"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    return render_template('upload_final.html')

@app.route("/about", methods=['GET', 'POST'])
def about():
	return render_template('about.html')

@app.route("/success", methods=['POST'])  
def success():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('no file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        print('data/files/'+file.filename)
        filename = secure_filename(file.filename)
        file.save('data/files/'+file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        import predict as pr
        pr.getfile(file.filename)
        image_path = 'data/rib_suppression/' + file.filename
        image1 = image.load_img(image_path, target_size=(256, 256))
        img_data = np.expand_dims(image1, axis=0)
        print(img_data.shape)
        with graph.as_default():
            out = model.predict(img_data)
            print(out)
            y_classes = out.argmax(axis=-1)
            print(y_classes[0])
            class_map = {0:'COVID-19', 1:'Normal', 2:'Pneumonia'}
            predicted_value = class_map[y_classes[0]]
            print(predicted_value)
        print("done")
        flash("Image uploded successfully")
        return render_template("success.html", name = filename, 
            predicted_value=predicted_value)
            # image='static/uploads/'+filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)


if __name__ == '__main__':
	app.run(debug=True)