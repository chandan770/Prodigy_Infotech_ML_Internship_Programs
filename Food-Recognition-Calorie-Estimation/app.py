import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2
from safetensors.torch import load_file

from PIL import Image, ImageTk
from huggingface_hub import hf_hub_download


# ============================================================
# PROJECT
# ============================================================

PROJECT_PATH = r"E:\Produgy_Infotech_ML_Internship_Projects\task 5"


# ============================================================
# FOOD-101 MODEL
# ============================================================

MODEL_REPO = "Lumia101/Food101-EfficientNet-B0"

print("Loading Food-101 model...")

config_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="config.json"
)

weights_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model.safetensors"
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(config_path, "r") as f:
    config = json.load(f)

id2label = config["id2label"]


# ============================================================
# CREATE MODEL
# ============================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier = nn.Sequential(
    nn.Dropout(
        p=0.2,
        inplace=True
    ),

    nn.Linear(
        1280,
        512
    ),

    nn.SiLU(),

    nn.Dropout(
        p=0.2
    ),

    nn.Linear(
        512,
        101
    )
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

state_dict = load_file(
    weights_path
)

model.load_state_dict(
    state_dict
)

model.eval()

print("Food-101 model loaded successfully!")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = v2.Compose([
    v2.Resize(160),

    v2.CenterCrop(128),

    v2.ToImage(),

    v2.ToDtype(
        torch.float32,
        scale=True
    ),

    v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# CALORIE DATABASE
# ============================================================
# Approximate kcal per 100g.
# These values are reference values, not image-measured values.

CALORIE_DATA = {

    "apple_pie": 237,
    "baby_back_ribs": 320,
    "baklava": 428,
    "beef_carpaccio": 120,
    "beef_tartare": 150,
    "beet_salad": 70,
    "beignets": 300,
    "bibimbap": 130,
    "bread_pudding": 200,
    "breakfast_burrito": 220,
    "bruschetta": 150,
    "caesar_salad": 180,
    "cannoli": 280,
    "caprese_salad": 150,
    "carrot_cake": 400,
    "ceviche": 100,
    "cheesecake": 321,
    "cheese_plate": 350,
    "chicken_curry": 190,
    "chicken_quesadilla": 230,
    "chicken_wings": 290,
    "chocolate_cake": 370,
    "chocolate_mousse": 225,
    "churros": 400,
    "clam_chowder": 95,
    "club_sandwich": 250,
    "crab_cakes": 220,
    "creme_brulee": 250,
    "croque_madame": 250,
    "cup_cakes": 350,
    "deviled_eggs": 145,
    "donuts": 430,
    "dumplings": 220,
    "edamame": 121,
    "eggs_benedict": 220,
    "escargots": 90,
    "falafel": 333,
    "filet_mignon": 250,
    "fish_and_chips": 220,
    "foie_gras": 460,
    "french_fries": 312,
    "french_onion_soup": 75,
    "french_toast": 229,
    "fried_calamari": 175,
    "fried_rice": 163,
    "frozen_yogurt": 127,
    "garlic_bread": 350,
    "gnocchi": 150,
    "greek_salad": 100,
    "grilled_cheese_sandwich": 350,
    "grilled_salmon": 208,
    "guacamole": 150,
    "gyoza": 200,
    "hamburger": 295,
    "hot_and_sour_soup": 40,
    "hot_dog": 290,
    "huevos_rancheros": 150,
    "hummus": 166,
    "ice_cream": 207,
    "lasagna": 135,
    "lobster_bisque": 85,
    "lobster_roll_sandwich": 220,
    "macaroni_and_cheese": 164,
    "macarons": 400,
    "mussels": 86,
    "nachos": 350,
    "omelette": 154,
    "onion_rings": 320,
    "oysters": 68,
    "pad_thai": 190,
    "paella": 160,
    "pancakes": 227,
    "panna_cotta": 250,
    "peking_duck": 337,
    "pho": 60,
    "pizza": 266,
    "pork_chop": 231,
    "poutine": 310,
    "prime_rib": 300,
    "pulled_pork_sandwich": 250,
    "ramen": 90,
    "ravioli": 180,
    "red_velvet_cake": 350,
    "risotto": 170,
    "samosa": 260,
    "sashimi": 140,
    "scallops": 111,
    "seaweed_salad": 70,
    "shrimp_and_grits": 170,
    "spaghetti_bolognese": 160,
    "spaghetti_carbonara": 190,
    "spring_rolls": 200,
    "steak": 271,
    "strawberry_shortcake": 250,
    "sushi": 150,
    "tacos": 226,
    "takoyaki": 180,
    "tiramisu": 240,
    "tuna_tartare": 150,
    "waffles": 291,
    "waffles_with_fruit": 200
}


# ============================================================
# GLOBAL VARIABLES
# ============================================================

selected_image_path = None

prediction_results = []


# ============================================================
# FIND CALORIES
# ============================================================

def get_calories(food_name):

    food_name = food_name.lower().replace(" ", "_")

    if food_name in CALORIE_DATA:
        return CALORIE_DATA[food_name]

    return None


# ============================================================
# SELECT IMAGE
# ============================================================

def select_image():

    global selected_image_path

    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select Food Image",
        filetypes=[
            (
                "Image Files",
                "*.jpg *.jpeg *.jfif *.png *.webp *.bmp"
            ),
            (
                "All Files",
                "*.*"
            )
        ]
    )

    if not file_path:
        return

    selected_image_path = file_path

    try:

        image = Image.open(
            file_path
        ).convert("RGB")

        display_image = image.copy()

        display_image.thumbnail(
            (430, 320)
        )

        photo = ImageTk.PhotoImage(
            display_image
        )

        image_label.config(
            image=photo,
            text=""
        )

        image_label.image = photo

        # Reset results
        food_result.config(
            text="Food: --"
        )

        confidence_result.config(
            text="Confidence: --"
        )

        calorie_result.config(
            text="Calories / 100g: --"
        )

        estimated_result.config(
            text="Estimated Calories: --"
        )

        top_predictions.config(
            text="Top Predictions:\n--"
        )

    except Exception as e:

        messagebox.showerror(
            "Image Error",
            str(e)
        )


# ============================================================
# PREDICT FOOD
# ============================================================

def predict_food():

    global prediction_results

    if selected_image_path is None:

        messagebox.showwarning(
            "No Image",
            "Please select a food image first."
        )

        return

    try:

        # Load image
        image = Image.open(
            selected_image_path
        ).convert("RGB")

        # Apply exact model preprocessing
        image_tensor = transform(
            image
        ).unsqueeze(0)

        # Prediction
        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            top = torch.topk(
                probabilities,
                5
            )

        prediction_results = []

        for index, probability in zip(
            top.indices[0],
            top.values[0]
        ):

            label = id2label[
                str(index.item())
            ]

            confidence = (
                probability.item() * 100
            )

            prediction_results.append(
                (label, confidence)
            )

        # Best prediction
        food = prediction_results[0][0]
        confidence = prediction_results[0][1]

        # Display main prediction
        food_result.config(
            text=f"Food: {food.replace('_', ' ').title()}"
        )

        confidence_result.config(
            text=f"Confidence: {confidence:.2f}%"
        )

        # Top 5
        top_text = "Top Predictions:\n\n"

        for i, (label, score) in enumerate(
            prediction_results,
            start=1
        ):

            top_text += (
                f"{i}. "
                f"{label.replace('_', ' ').title()}"
                f" — {score:.2f}%\n"
            )

        top_predictions.config(
            text=top_text
        )

        # Calories
        calories = get_calories(
            food
        )

        if calories is not None:

            calorie_result.config(
                text=(
                    f"Calories / 100g: "
                    f"{calories} kcal"
                )
            )

            calculate_calories()

        else:

            calorie_result.config(
                text="Calories / 100g: Not available"
            )

            estimated_result.config(
                text="Estimated Calories: --"
            )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ============================================================
# CALCULATE CALORIES
# ============================================================

def calculate_calories():

    if not prediction_results:
        return

    try:

        portion = float(
            portion_entry.get()
        )

        food = prediction_results[0][0]

        calories = get_calories(
            food
        )

        if calories is None:

            estimated_result.config(
                text="Estimated Calories: N/A"
            )

            return

        estimated = (
            calories * portion / 100
        )

        estimated_result.config(
            text=(
                f"Estimated Calories: "
                f"{estimated:.1f} kcal"
            )
        )

    except ValueError:

        messagebox.showwarning(
            "Invalid Portion",
            "Please enter a valid number of grams."
        )


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "AI Food Recognition & Calorie Estimator"
)

root.geometry(
    "1050x750"
)

root.resizable(
    False,
    False
)

root.configure(
    bg="#f4f6f8"
)


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg="#1f2937",
    height=100
)

header.pack(
    fill="x"
)

title = tk.Label(
    header,
    text="🍽️ AI FOOD RECOGNITION",
    font=("Arial", 25, "bold"),
    fg="white",
    bg="#1f2937"
)

title.pack(
    pady=(18, 2)
)

subtitle = tk.Label(
    header,
    text="Food Classification & Calorie Estimation",
    font=("Arial", 12),
    fg="#d1d5db",
    bg="#1f2937"
)

subtitle.pack()


# ============================================================
# MAIN CONTENT
# ============================================================

main_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=35,
    pady=25
)


# ============================================================
# IMAGE CARD
# ============================================================

image_card = tk.Frame(
    main_frame,
    bg="white",
    width=500,
    height=470
)

image_card.pack(
    side="left",
    padx=15
)

image_card.pack_propagate(False)


image_title = tk.Label(
    image_card,
    text="Food Image",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="#111827"
)

image_title.pack(
    pady=(20, 10)
)


image_label = tk.Label(
    image_card,
    text="📷\n\nNo image selected",
    font=("Arial", 16),
    bg="white",
    fg="#6b7280"
)

image_label.pack(
    expand=True
)


# ============================================================
# RESULT CARD
# ============================================================

result_card = tk.Frame(
    main_frame,
    bg="white",
    width=450,
    height=470
)

result_card.pack(
    side="right",
    padx=15
)

result_card.pack_propagate(False)


result_title = tk.Label(
    result_card,
    text="Prediction Result",
    font=("Arial", 21, "bold"),
    bg="white",
    fg="#111827"
)

result_title.pack(
    pady=(20, 15)
)


food_result = tk.Label(
    result_card,
    text="Food: --",
    font=("Arial", 16, "bold"),
    bg="white",
    fg="#111827"
)

food_result.pack(
    pady=5
)


confidence_result = tk.Label(
    result_card,
    text="Confidence: --",
    font=("Arial", 14),
    bg="white",
    fg="#2563eb"
)

confidence_result.pack(
    pady=5
)


calorie_result = tk.Label(
    result_card,
    text="Calories / 100g: --",
    font=("Arial", 14),
    bg="white",
    fg="#374151"
)

calorie_result.pack(
    pady=5
)


# ============================================================
# TOP PREDICTIONS
# ============================================================

top_predictions = tk.Label(
    result_card,
    text="Top Predictions:\n--",
    justify="left",
    font=("Arial", 11),
    bg="white",
    fg="#4b5563"
)

top_predictions.pack(
    pady=10
)


# ============================================================
# PORTION
# ============================================================

portion_label = tk.Label(
    result_card,
    text="Portion Size (grams)",
    font=("Arial", 12, "bold"),
    bg="white"
)

portion_label.pack(
    pady=(5, 3)
)


portion_entry = tk.Entry(
    result_card,
    font=("Arial", 13),
    width=12,
    justify="center"
)

portion_entry.insert(
    0,
    "100"
)

portion_entry.pack(
    pady=3
)


calculate_button = tk.Button(
    result_card,
    text="Calculate Calories",
    command=calculate_calories,
    font=("Arial", 10, "bold"),
    bg="#f59e0b",
    fg="white",
    padx=15,
    pady=7,
    relief="flat"
)

calculate_button.pack(
    pady=7
)


estimated_result = tk.Label(
    result_card,
    text="Estimated Calories: --",
    font=("Arial", 15, "bold"),
    bg="white",
    fg="#dc2626"
)

estimated_result.pack(
    pady=5
)


# ============================================================
# BUTTONS
# ============================================================

button_frame = tk.Frame(
    root,
    bg="#f4f6f8"
)

button_frame.pack(
    pady=10
)


select_button = tk.Button(
    button_frame,
    text="📁 Select Food Image",
    command=select_image,
    font=("Arial", 13, "bold"),
    bg="#2563eb",
    fg="white",
    padx=25,
    pady=12,
    relief="flat"
)

select_button.pack(
    side="left",
    padx=10
)


predict_button = tk.Button(
    button_frame,
    text="🔍 Predict Food",
    command=predict_food,
    font=("Arial", 13, "bold"),
    bg="#16a34a",
    fg="white",
    padx=25,
    pady=12,
    relief="flat"
)

predict_button.pack(
    side="left",
    padx=10
)


# ============================================================
# FOOTER
# ============================================================

footer = tk.Label(
    root,
    text="Food-101 • EfficientNet-B0 • PyTorch",
    font=("Arial", 9),
    bg="#f4f6f8",
    fg="#6b7280"
)

footer.pack(
    pady=(0, 8)
)


# ============================================================
# START
# ============================================================

root.mainloop()