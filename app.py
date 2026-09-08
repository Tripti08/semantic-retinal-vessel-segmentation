import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from UNET.model import UNet


MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best_model.pth"))
IMAGE_SIZE = 512


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device


def predict(model, device, image):
    original_size = image.size
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (prediction > 0.5).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)
    return mask_image


st.set_page_config(page_title="Retinal Vessel Segmentation", page_icon="🩺", layout="wide")
st.title("Retinal Blood Vessel Segmentation")
st.write("Upload a retinal fundus image to generate a vessel segmentation mask.")

loaded = load_model()
if loaded is None:
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.info("Train the model first, then place best_model.pth in the checkpoints/ folder.")
    st.stop()

model, device = loaded
uploaded = st.file_uploader("Upload retinal image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    mask = predict(model, device, image)
    image_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    overlay = image_array.copy()
    vessel_pixels = mask_array > 0
    overlay[vessel_pixels] = [255, 0, 0]

    left, middle, right = st.columns(3)
    left.image(image, caption="Input image", use_container_width=True)
    middle.image(mask, caption="Predicted vessels", use_container_width=True)
    right.image(overlay, caption="Overlay", use_container_width=True)

