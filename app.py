import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Pro AI OCR Scanner", layout="wide")

# 2. Header
st.title("📄 Pro AI Document OCR Scanner")
st.markdown("---")

# 3. Sidebar - Advanced Settings
st.sidebar.header("OCR Settings")

# FEATURE: Multi-Language Support
language_options = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Dutch": "nl"
}

selected_lang_names = st.sidebar.multiselect(
    "Select Languages",
    options=list(language_options.keys()),
    default=["English"]
)
selected_langs = [language_options[name] for name in selected_lang_names]


@st.cache_resource
def load_model(langs):
    return easyocr.Reader(list(langs))


reader = load_model(tuple(selected_langs))

# FEATURE: Image Enhancement (The "Secret Sauce")
st.sidebar.markdown("---")
st.sidebar.subheader("Image Enhancement")
enhance_image = st.sidebar.checkbox("Enhance for Accuracy", value=True)

# 4. Sidebar for Uploading
uploaded_file = st.sidebar.file_uploader(
    "Upload an Image", type=["jpg", "jpeg", "png"])

# 5. Main Logic
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Process image for the AI
    if enhance_image:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        img_processed = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        img_processed = img_array

    with col1:
        st.subheader("Visual Preview")
        st.image(img_processed, use_container_width=True,
                 caption="What the AI is reading")

    with col2:
        st.subheader("Extracted Results")
        with st.spinner("AI is analyzing..."):
            results = reader.readtext(img_processed, detail=0)
            final_text = "\n".join([str(res) for res in results])

            if final_text:
                # NEW FEATURE: Search within scanned text
                search_term = st.text_input("🔍 Search for a word in results:")
                if search_term:
                    if search_term.lower() in final_text.lower():
                        st.success(f"Found '{search_term}' in the document!")
                    else:
                        st.error(f"'{search_term}' not found.")

                st.text_area(label="Result:", value=final_text, height=350)
                st.download_button("📥 Download Text", final_text,
                                   file_name="scanned_text.txt")
            else:
                st.warning("No text detected. Try enabling enhancement.")
else:
    st.info("Upload an image to start scanning.")
