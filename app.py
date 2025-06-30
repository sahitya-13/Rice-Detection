from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice.h5')
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  # match exactly
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(path)
        processed_img = prepare_image(path)
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        return render_template('results.html', prediction=predicted_class, image_path=path)
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        img = request.files['image']
        path = os.path.join(UPLOAD_FOLDER, img.filename)
        img.save(path)
        processed_img = prepare_image(path)
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        return render_template('results.html', prediction=predicted_class, image_path=path)
    return render_template('details.html')

if __name__ == '__main__':
    app.run(debug=True)
