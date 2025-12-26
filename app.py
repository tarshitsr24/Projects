import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# --- PERMANENT FIX: CACHE THE MODEL ---
# This ensures the 100MB+ AI model only loads ONCE, preventing memory crashes.
@st.cache_resource
def get_reader():
    # gpu=False is mandatory for free cloud servers.
    return easyocr.Reader(['en'], gpu=False)

def main():
    st.set_page_config(page_title="Pro AI OCR Scanner", layout="wide")
    st.title("📄 Pro AI Document OCR Scanner")

    # Load the cached reader
    reader = get_reader()

    # Sidebar for controls
    st.sidebar.header("OCR Settings")
    enhance = st.sidebar.checkbox("Enhance for Accuracy", value=True)
    
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visual Preview")
            # Apply OpenCV enhancement if selected
            if enhance:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            st.image(img_array, use_container_width=True)

        with col2:
            st.subheader("Extracted Results")
            if st.button("Start Scan"):
                with st.spinner("AI is reading..."):
                    # Perform OCR
                    results = reader.readtext(img_array, detail=0)
                    text_output = "\n".join(results)
                    
                    if text_output:
                        st.text_area("Result:", value=text_output, height=300)
                        st.download_button("📥 Download Text", text_output, file_name="scan.txt")
                    else:
                        st.warning("No text detected.")

if __name__ == "__main__":
    main()
