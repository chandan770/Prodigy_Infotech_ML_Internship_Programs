import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = r"E:\Produgy_Infotech_ML_Internship_Projects\task 5"

DATASET_DIR = DATASET_DIR = r"E:\Produgy_Infotech_ML_Internship_Projects\task 5\data\Food Classification dataset"

MODEL_DIR = os.path.join(PROJECT_PATH, "model")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("FOOD IMAGE CLASSIFICATION")
print("=" * 60)

print("Dataset path:")
print(DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(
        f"\nDataset not found!\n\nExpected location:\n{DATASET_DIR}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# Save class names
class_names_path = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=4)

print("\nClass names saved to:")
print(class_names_path)


# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# ============================================================
# MOBILE NET V2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "food_recognition_model.keras"
)

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print("\nValidation Accuracy:")
print(f"{accuracy * 100:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

model.save(model_path)

print("\nModel saved successfully:")
print(model_path)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()
plt.grid()

accuracy_graph = os.path.join(
    MODEL_DIR,
    "accuracy.png"
)

plt.savefig(accuracy_graph)

plt.show()


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid()

loss_graph = os.path.join(
    MODEL_DIR,
    "loss.png"
)

plt.savefig(loss_graph)

plt.show()


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)