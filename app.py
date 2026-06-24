# app.py
# Streamlit front-end for the Fruit Freshness Detector.
# Lets a user upload (or pick) a fruit photo and shows whether the model
# thinks it is Fresh or Rotten, which fruit it is, and how confident it is.
#
# Run with:  streamlit run app.py
# Streamlit docs: https://docs.streamlit.io

import os
from PIL import Image
import streamlit as st

# predict_image() is the model wrapper written by the team (predict.py).
# It returns a (state, fruit, confidence) tuple for a PIL image.
from predict import predict_image

# Folder of example pictures shipped with the project.
SAMPLE_DIR = "sample_images"


def list_sample_images():
    """Return the sample image filenames so the user can try the app
    without having a photo of their own."""
    if not os.path.isdir(SAMPLE_DIR):
        return []
    allowed = (".jpg", ".jpeg", ".png", ".webp")
    return [f for f in sorted(os.listdir(SAMPLE_DIR)) if f.lower().endswith(allowed)]


def show_result(image):
    """Run the model on one PIL image and present the outcome nicely."""
    # Ask the model. predict_image does the resizing/normalising internally.
    state, fruit, confidence = predict_image(image)

    fruit = fruit.strip().capitalize()        # "apples" -> "Apples"
    confidence = round(confidence, 1)

    # Pick wording and colour based on the verdict.
    if state == "Fresh":
        st.success(f"✅ This {fruit} looks **FRESH**")
        advice = "Good to eat. Enjoy it!"
    else:
        st.error(f"🦠 This {fruit} looks **ROTTEN**")
        advice = "Better not eat this one — throw it away."

    # Two small metric boxes side by side.
    left, right = st.columns(2)
    left.metric("Fruit", fruit)
    right.metric("Verdict", state)

    # A confidence read-out plus a bar (0.0–1.0).
    st.write(f"**Model confidence:** {confidence}%")
    st.progress(min(confidence / 100, 1.0))
    st.caption(advice)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")

st.title("🍎 Fruit Freshness Detector")
st.write(
    "Upload a photo of an **apple, banana, or orange** and the model will "
    "guess whether it is fresh or rotten. You can also try one of the sample "
    "images below."
)

# Let the user choose how to provide an image.
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

# Once we have an image, show it and offer the analyse button.
if image is not None:
    st.image(image, caption="Selected image", use_container_width=True)

    if st.button("Analyse 🔍"):
        with st.spinner("Looking at your fruit..."):
            show_result(image)
else:
    st.info("Waiting for an image...")

# Small footer.
st.markdown("---")
st.caption("AIT102 Group Project · Model: MobileNetV2 · Front-end: Streamlit")
