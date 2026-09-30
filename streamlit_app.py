
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Digit Classifier", page_icon="🔢")

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

@st.cache_resource
def load_model():
    model = DigitCNN()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

st.title("🔢 Handwritten Digit Image Classifier")
st.write("Upload an image containing a handwritten digit from 0 to 9.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    # Preprocessing: grayscale, invert if needed, resize to 8x8,
    # then scale pixels to the same range used during training.
    image = ImageOps.autocontrast(image)

    # Make the digit white on a black background when appropriate.
    arr = np.array(image)
    if arr.mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((8, 8))
    image_array = np.array(image).astype("float32") / 255.0

    tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities))
        confidence = float(probabilities[prediction])

    st.success(f"Prediction: **{prediction}**")
    st.metric("Confidence", f"{confidence:.2%}")

    st.subheader("Class Probabilities")
    chart_data = {
        str(i): float(probabilities[i])
        for i in range(10)
    }
    st.bar_chart(chart_data)
else:
    st.info("Upload a handwritten digit image to get a prediction.")
