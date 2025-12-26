# 📄 Pro AI Document OCR Scanner

A high-performance Optical Character Recognition (OCR) web application built with Python. This tool leverages Deep Learning to extract text from images, supporting multi-language scanning and advanced image pre-processing for maximum accuracy.

## 🚀 Key Features

* **Deep Learning Engine**: Powered by **EasyOCR**, utilizing a CRAFT text detector and an LSTM recognition model.
* **Image Enhancement Pipeline**: Integrated **OpenCV** filters (Grayscale & OTSU Thresholding) to clean stylized or low-contrast images before scanning.
* **Multi-Language Support**: Simultaneous text extraction for English, German, French, Spanish, Italian, and Dutch.
* **Interactive Search**: Real-time keyword searching within the extracted text results.
* **Streamlit UI**: A clean, responsive dashboard for easy file uploads and live previews.

## 🛠️ Built With

* **Python 3.13+**: Utilizing the latest performance improvements.
* **Streamlit**: For the web interface.
* **EasyOCR**: For the AI/Deep Learning logic.
* **OpenCV**: For computer vision and image pre-processing.
* **NumPy & Pillow**: For efficient array and image handling.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/python-ocr-app.git](https://github.com/yourusername/python-ocr-app.git)
   cd python-ocr-app

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   source .venv/bin/activate # Mac/Linux

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the application:**
   ```bash
   streamlit run app.py

