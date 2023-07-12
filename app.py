import os

import tensorflow as tf
from flask import Flask, request, render_template

from classifier import classify

STATIC_FOLDER = "static/"
TEMPLATES_FOLDER = "templates/"


app = Flask(__name__)

UPLOAD_FOLDER = STATIC_FOLDER + "upload/"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "pepsi_coca.keras",
    compile=False,
)


@app.get("/")
def home():
    return render_template("home.html")


@app.post("/")
def upload_image():
    try:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)
        label, prob = classify(cnn_model, upload_image_path)

        return render_template("classification.html", label=label, prob=prob, upload_image_path=upload_image_path)

    except ValueError:
        error_message = "Invalid image file format. Supported formats are JPEG, PNG, GIF, TIFF."
        return render_template("error.html", error_message=error_message)
