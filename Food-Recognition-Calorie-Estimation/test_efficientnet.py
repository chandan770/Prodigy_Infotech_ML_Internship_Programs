import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image


# ============================================================
# DOWNLOAD MODEL FILES
# ============================================================

print("Downloading Food-101 EfficientNet model...")

config_path = hf_hub_download(
    repo_id="Lumia101/Food101-EfficientNet-B0",
    filename="config.json"
)

weights_path = hf_hub_download(
    repo_id="Lumia101/Food101-EfficientNet-B0",
    filename="model.safetensors"
)

print("Model downloaded successfully!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(config_path, "r") as f:
    config = json.load(f)

id2label = config["id2label"]


# ============================================================
# CREATE EFFICIENTNET MODEL
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

print("Loading model weights...")

state_dict = load_file(weights_path)

model.load_state_dict(
    state_dict
)

model.eval()

print("Model loaded successfully!")


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
# IMAGE PATH
# ============================================================

IMAGE_PATH = r"E:\Produgy_Infotech_ML_Internship_Projects\task 5\pizza.jpg.jfif"


# ============================================================
# PREDICTION
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(
    image
).unsqueeze(0)


print("Predicting...")

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


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FOOD-101 PREDICTION")
print("=" * 60)

for index, probability in zip(
    top.indices[0],
    top.values[0]
):

    label = id2label[str(index.item())]

    confidence = probability.item() * 100

    print(
        f"{label:30s} {confidence:.2f}%"
    )

print("=" * 60)