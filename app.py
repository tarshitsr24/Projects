import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# --- STABLE CACHING FOR THE AI MODEL ---
# This ensures the model only loads once and stays in memory.
@st.cache_resource
def get_reader():
    # Only loading English ('en') to maximize stability.
    return easyocr.Reader(['en'], gpu=False)

def main():
    st.set_page_config(page_title="Pro AI OCR Scanner", layout="wide")
    st.title("📄 Pro AI Document OCR Scanner")

    # Sidebar: Core Features
    st.sidebar.header("1. Image Processing")
    enhance = st.sidebar.toggle("Enhance Image (Grayscale + Otsu)", value=True)
    
    st.sidebar.header("2. Upload Document")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    # Initialize the stable English reader
    reader = get_reader()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visual Preview")
            display_img = img_array.copy()
            
            # Apply Image Enhancement logic
            if enhance:
                # Handle images with Alpha channels (RGBA to RGB)
                if len(display_img.shape) == 3 and display_img.shape[2] == 4:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGBA2RGB)
                
                # Convert to Grayscale and apply Otsu's Thresholding
                gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
                display_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("Extracted Results")
            if st.button("🚀 Start AI Scan"):
                with st.spinner("AI is reading English text..."):
                    # Use the processed image for better OCR results
                    results = reader.readtext(display_img if enhance else img_array, detail=0)
                    st.session_state['full_text'] = "\n".join(results)
            
            # Keyword Search & Download Results
            if 'full_text' in st.session_state:
                search_term = st.text_input("🔍 Search inside results:", placeholder="Type a word...")
                
                display_text = st.session_state['full_text']
                if search_term:
                    if search_term.lower() in display_text.lower():
                        st.success(f"Found: '{search_term}'")
                    else:
                        st.error(f"'{search_term}' not found.")
                
                st.text_area("Final Text Output:", value=display_text, height=300)
                st.download_button("📥 Download Result", display_text, file_name="ocr_result.txt")

if __name__ == "__main__":
    main()
