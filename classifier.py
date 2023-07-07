import tensorflow as tf

IMAGE_SIZE = (300, 300)


def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def load_and_preprocess_image(image_path: str):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMAGE_SIZE
    )

    return preprocess_image(image)


def classify(model, image_path: str):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)

    score = predictions[0][0]

    label = "Pepsi" if score <= 0.5 else "Coca-Cola"
    prob = 1 - score if label == "Pepsi" else score
    prob = round(prob * 100, 2)

    return label, prob
