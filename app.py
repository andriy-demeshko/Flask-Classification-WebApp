import os

import tensorflow as tf
from flask import Flask, request, render_template

from classifier import classify

STATIC_FOLDER = "static/"
TEMPLATES_FOLDER = "templates/"


app = Flask(__name__)

UPLOAD_FOLDER = STATIC_FOLDER + "upload/"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "pepsi_coca",
    compile=False,
)


@app.get("/")
def home():
    return render_template("home.html")


@app.post("/")
def upload_image():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    return render_template("classification.html", label=label, prob=prob)
