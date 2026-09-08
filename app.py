import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import torch
from PIL import Image

from UNET.model import UNet


MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best_model.pth"))
CHECKPOINT_ID = "1Wl7-E6Tk3YpeJ7GIYScGvUeW9ou474yy"
IMAGE_SIZE = 512


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=CHECKPOINT_ID, output=str(MODEL_PATH), quiet=False)
        except Exception as error:
            st.error(f"Could not download the model checkpoint: {error}")
            return None
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
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
    return Image.fromarray(mask).resize(original_size, Image.Resampling.NEAREST)


st.set_page_config(page_title="Retinal Vessel Segmentation", page_icon="🩺", layout="wide")
st.title("Retinal Blood Vessel Segmentation")
st.write("Upload a retinal fundus image to generate a vessel segmentation mask.")

loaded = load_model()
if loaded is None:
    st.error("The model checkpoint is unavailable.")
    st.info("The app downloads the pretrained U-Net checkpoint automatically. Check the deployment logs if the download failed.")
    st.stop()

model, device = loaded
uploaded = st.file_uploader("Upload retinal image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    mask = predict(model, device, image)
    image_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask, dtype=np.uint8)
    overlay = image_array.copy()
    overlay[mask_array > 0] = [255, 0, 0]

    left, middle, right = st.columns(3)
    left.image(image, caption="Input image", use_container_width=True)
    middle.image(mask, caption="Predicted vessels", use_container_width=True)
    right.image(overlay, caption="Overlay", use_container_width=True)
