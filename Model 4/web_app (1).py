import streamlit as st
import fitz
import pytesseract
from PIL import Image
import io

# Setup Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI PDF Reader", layout="centered")

st.title("📄 AI PDF Text Extractor")
st.write("Upload a PDF file (Text or Scanned) to extract its content.")

# Uploading file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Processing... Please wait.'):
        # Read the PDF from memory
        pdf_data = uploaded_file.read()
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                full_text += f"\n--- Page {page_num+1} ---\n{text}"
            else:
                # OCR for images
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img, lang='eng+ara')
                full_text += f"\n--- Page {page_num+1} (OCR) ---\n{ocr_text}"
        
        st.success("Done!")
        
        # Display results
        st.text_area("Extracted Text:", value=full_text, height=400)
        
        # Download button
        st.download_button("Download Text File", full_text, file_name="extracted_text.txt")