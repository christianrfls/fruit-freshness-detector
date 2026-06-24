# evaluate_model.py: test set accuracy and per-class breakdown
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Image loading approach adapted from the TensorFlow image classification tutorial:
# https://www.tensorflow.org/tutorials/images/classification
test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
class_names = test_data.class_names

# Loading the saved model from the Keras save and load guide:
# https://www.tensorflow.org/tutorials/keras/save_and_load
model = tf.keras.models.load_model("model/freshness_model.keras")

# Overall score.
loss, accuracy = model.evaluate(test_data)
print("Test accuracy:", round(accuracy * 100, 2), "%")

# FLAG: Prediction loop collecting true and predicted labels closely follows
# the evaluation pattern from the TensorFlow image classification tutorial:
# https://www.tensorflow.org/tutorials/images/classification
# Detailed score per class.
true_labels = []
pred_labels = []
for images, labels in test_data:
    preds = model.predict(images)
    true_labels.extend(np.argmax(labels, axis=1))
    pred_labels.extend(np.argmax(preds, axis=1))

# Classification report and confusion matrix from scikit-learn metrics documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print(classification_report(true_labels, pred_labels, target_names=class_names))
print(confusion_matrix(true_labels, pred_labels))