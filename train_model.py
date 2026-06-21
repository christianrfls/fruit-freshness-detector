import tensorflow as tf

IMG_SIZE = (224, 224)   # MobileNetV2 expects 224x224
BATCH_SIZE = 32         # images per training step

# folder names become the class labels
train_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_data.class_names   # needed later in the app
print("Classes found:", class_names)

# keep images ready in memory so training is not slowed by disk reads.
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)

# MobileNetV2 pretrained on ImageNet; drop its classification head, add our own
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # keep its weights as-is

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1),  # scale pixels to the range MobileNet expects
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),           # shrink the output to a simple list of numbers
    tf.keras.layers.Dropout(0.2),                       # helps prevent memorizing instead of learning
    tf.keras.layers.Dense(6, activation="softmax")      # 6 outputs, one per class, as probabilities
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()   # prints the structure 

EPOCHS = 5
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

model.save("model/freshness_model.keras")   # so the app doesn't need to retrain everytime
print("Model saved to model/freshness_model.keras")

with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")
print("Class names saved.")