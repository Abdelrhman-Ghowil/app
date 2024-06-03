import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import DetrImageProcessor, DetrForSegmentation

# Load model and processor
processor = DetrImageProcessor.from_pretrained("briaai/RMBG-1.4")
model = DetrForSegmentation.from_pretrained("briaai/RMBG-1.4")

st.title("Background Remover App")
st.write("Upload an image to remove its background")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # Get masks
    mask = logits.argmax(-1).squeeze().detach().cpu().numpy()

    # Create the segmentation result
    result = Image.fromarray((mask * 255).astype(np.uint8))

    # Display the result
    st.image(result, caption='Background Removed Image', use_column_width=True)
