# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/freshness_model.keras")
with open("model/class_names.txt") as f:
    class_names = [line.strip() for line in f]

def predict_image(image):
    # resize and normalize before feeding to model
    image = image.convert("RGB").resize((224, 224))
    array = np.array(image)
    array = np.expand_dims(array, axis=0)   # add a batch dimension

    predictions = model.predict(array)[0]
    best = np.argmax(predictions)
    label = class_names[best]
    confidence = float(predictions[best]) * 100

    # Turn "freshapples" into something friendly.
    state = "Fresh" if label.startswith("fresh") else "Rotten"
    fruit = label.replace("fresh", "").replace("rotten", "")
    return state, fruit, confidence

# Quick test when run directly.
if __name__ == "__main__":
    test = Image.open("sample_images/banana_citrus_2.jpeg")
    print(predict_image(test))