import tensorflow as tf

IMAGE_SIZE = (150, 150)


def load_and_preprocess_image(image_path: str):
    try:
        image = tf.keras.preprocessing.image.load_img(
            image_path, target_size=IMAGE_SIZE
        )
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        return img_array

    except Exception as e:
        raise ValueError("Invalid image file format. Supported formats are JPEG, PNG, GIF, TIFF.") from e


def classify(model, image_path: str):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)

    score = predictions[0][0]

    label = "Coca-Cola" if score <= 0.5 else "Pepsi"
    prob = f"{(1 - score) * 100} %" if label == "Coca-Cola" else f"{score * 100} %"

    return label, prob
