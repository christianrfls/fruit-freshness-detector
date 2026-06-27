# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image

# Loading the saved model from:
# https://www.tensorflow.org/tutorials/keras/save_and_load
model = tf.keras.models.load_model("model/freshness_model.keras")
with open("model/class_names.txt") as f:
    class_names = [line.strip() for line in f]

def predict_image(image):
    # Image preprocessing (resize, array conversion, batch dimension) and prediction pattern adapted from:
    # https://www.tensorflow.org/tutorials/images/classification
    
    # resize and normalize image before feeding to model
    image = image.convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)   # add a batch dimension

    predictions = model.predict(array)[0]
    best = np.argmax(predictions)
    label = class_names[best]
    confidence = float(predictions[best]) * 100

    # Turn class names (e.g., "freshapples") to be more readable
    state = "Fresh" if label.startswith("fresh") else "Rotten"
    fruit = label.replace("fresh", "").replace("rotten", "")
    return state, fruit, confidence

# Quick test
if __name__ == "__main__":
    test = Image.open("sample_images/banana_citrus_2.jpeg")
    print(predict_image(test))