import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image

# --- STABLE CACHING FOR LANGUAGES ---
# This function only reruns if you change the language list, saving memory.
@st.cache_resource
def get_reader(langs):
    return easyocr.Reader(langs, gpu=False)

def main():
    st.set_page_config(page_title="Pro AI OCR Scanner", layout="wide")
    st.title("📄 Pro AI Document OCR Scanner")

    # --- SIDEBAR: RESTORED FEATURES ---
    st.sidebar.header("1. Settings")
    
    # Restored: Multiple Language Support
    lang_options = {
        "English": "en", "German": "de", "French": "fr", 
        "Spanish": "es", "Italian": "it", "Dutch": "nl"
    }
    selected_langs = st.sidebar.multiselect(
        "Select Languages", 
        options=list(lang_options.keys()), 
        default=["English"]
    )
    lang_codes = [lang_options[l] for l in selected_langs]
    
    # Restored: Image Enhancement (Grayscale/Binarization)
    st.sidebar.header("2. Image Processing")
    enhance = st.sidebar.toggle("Enhance Image (Grayscale + Otsu)", value=True)
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Document", type=['jpg', 'jpeg', 'png'])

    # Initialize the reader with chosen languages
    reader = get_reader(lang_codes)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Visual Preview")
            # Apply restored Grayscale/Otsu logic
            display_img = img_array.copy()
            if enhance:
                # If image is RGBA (4 channels), convert to RGB first
                if len(display_img.shape) == 3 and display_img.shape[2] == 4:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGBA2RGB)
                
                gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
                display_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("Extracted Results")
            if st.button("🚀 Start AI Scan"):
                with st.spinner("AI is analyzing document..."):
                    # Use the enhanced image for better OCR accuracy
                    results = reader.readtext(display_img if enhance else img_array, detail=0)
                    st.session_state['full_text'] = "\n".join(results)
            
            # Restored: Keyword Search & Results
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
