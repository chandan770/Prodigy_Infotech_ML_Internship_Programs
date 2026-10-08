import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_PATH = r"E:\Prodigy_Infotech_ML_Internship_Projects\task 5"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "food_recognition_model.keras"
)

CLASS_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "class_names.json"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(CLASS_PATH, "r") as f:

    class_names = json.load(f)


print("Model loaded successfully!")


# =========================================================
# PREDICT FOOD
# =========================================================

def predict_food(image_path):

    # Load image
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to array
    img_array = image.img_to_array(
        img
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    # Best prediction
    index = np.argmax(
        predictions
    )

    food = class_names[index]

    confidence = (
        float(predictions[index])
        * 100
    )

    return food, confidence


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            r'python src\predict.py "image_path"'
        )

        sys.exit()

    image_path = sys.argv[1]

    food, confidence = predict_food(
        image_path
    )

    print("\n" + "=" * 55)
    print("FOOD RECOGNITION RESULT")
    print("=" * 55)

    print(
        f"Food       : "
        f"{food.replace('_', ' ').title()}"
    )

    print(
        f"Confidence : "
        f"{confidence:.2f}%"
    )

    print("=" * 55)