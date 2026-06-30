# Run with:  streamlit run app.py

import os
from PIL import Image
import streamlit as st

# Model wrapper: returns (state, fruit, confidence).
from predict import predict_image

# Example pictures shipped with the project.
SAMPLE_DIR = "sample_images"


def list_sample_images():
    """Return sample image filenames."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    allowed = (".jpg", ".jpeg", ".png", ".webp")
    return [f for f in sorted(os.listdir(SAMPLE_DIR)) if f.lower().endswith(allowed)]


def show_result(image):
    """Run the model and show the outcome."""
    state, fruit, confidence = predict_image(image)

    fruit = fruit.strip().capitalize()
    confidence = round(confidence, 1)

    # Wording/colour based on the verdict.
    if state == "Fresh":
        st.success(f"This {fruit} looks **FRESH**")
        advice = "Good to eat. Enjoy it!"
    else:
        st.error(f"This {fruit} looks **ROTTEN**")
        advice = "Better not eat this one — throw it away."

    # Two metric boxes side by side.
    left, right = st.columns(2)
    left.metric("Fruit", fruit)
    right.metric("Verdict", state)

    # Confidence read-out plus a bar.
    st.write(f"**Model confidence:** {confidence}%")
    st.progress(min(confidence / 100, 1.0))
    st.caption(advice)


# Page layout
st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")

st.title("Fruit Freshness Detector")
st.write(
    "Upload a photo of an **apple, banana, or orange** and the model will "
    "guess whether it is fresh or rotten. You can also try one of the sample "
    "images below."
)

# Choose how to provide an image.
mode = st.radio("Choose an image source:", ["Upload my own", "Use a sample image"])

image = None

if mode == "Upload my own":
    uploaded = st.file_uploader(
        "Upload a fruit photo", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is not None:
        image = Image.open(uploaded)

else:
    samples = list_sample_images()
    if samples:
        choice = st.selectbox("Pick a sample image:", samples)
        image = Image.open(os.path.join(SAMPLE_DIR, choice))
    else:
        st.warning("No sample images found in the 'sample_images' folder.")

# Show the image and offer the analyse button.
if image is not None:
    st.image(image, caption="Selected image", use_container_width=True)

    if st.button("Analyse"):
        with st.spinner("Looking at your fruit..."):
            show_result(image)
else:
    st.info("Waiting for an image...")

# Footer.
st.markdown("---")
st.caption("AIT102 Group Project · Model: MobileNetV2 · Front-end: Streamlit")
