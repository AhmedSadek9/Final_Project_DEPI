import pytesseract
from PIL import Image

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test if the connection is working
try:
    # This will print the version of Tesseract if linked correctly
    version = pytesseract.get_tesseract_version()
    print(f"Connected successfully! Tesseract version: {version}")
except Exception as e:
    print(f"Connection failed. Error: {e}")

import fitz  # This is the library that was missing
import pytesseract
from PIL import Image
import io
import os

# Link to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_pdf_content(pdf_path):
    # Check if the file exists first
    if not os.path.exists(pdf_path):
        return f"Error: The file '{pdf_path}' was not found in the folder."

    doc = fitz.open(pdf_path)
    full_content = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if text.strip():
            full_content += f"\n--- Page {page_num + 1} (Direct Text) ---\n{text}"
        else:
            # If page is an image, use OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img, lang='eng+ara')
            full_content += f"\n--- Page {page_num + 1} (OCR/Image) ---\n{ocr_text}"

    return full_content

if __name__ == "__main__":
    # IMPORTANT: Change 'my_document.pdf' to the name of your actual PDF file
    target_file = "mariammmmmww.pdf" 
    
    try:
        result = extract_pdf_content(target_file)
        with open("result.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("Success! Content extracted to result.txt")
    except Exception as e:
        print(f"An error occurred: {e}")