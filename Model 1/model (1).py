import customtkinter as ctk
from tkinter import filedialog
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import threading

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
PATH_TO_POPPLER = r'C:\Users\hp\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin'

class PDFReaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("smart PDF reader")
        self.geometry("600x500")

        
        self.label = ctk.CTkLabel(self, text="Drag PDF", font=("Arial", 16))
        self.label.pack(pady=20)

        self.btn = ctk.CTkButton(self, text="choose the PDF", command=self.upload_file)
        self.btn.pack(pady=10)

        self.textbox = ctk.CTkTextbox(self, width=500, height=300)
        self.textbox.pack(pady=20)

        self.status_label = ctk.CTkLabel(self, text="", text_color="yellow")
        self.status_label.pack()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.textbox.delete("0.0", "end")
            self.status_label.configure(text="Wait..")
           
            threading.Thread(target=self.process_pdf, args=(file_path,), daemon=True).start()

    def process_pdf(self, pdf_path):
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        full_text += f"--- صفحة {i+1} (نص) ---\n{text}\n\n"
                    else:
                        images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, poppler_path=PATH_TO_POPPLER)
                        for img in images:
                            ocr_text = pytesseract.image_to_string(img, lang='ara+eng')
                            full_text += f"--- صفحة {i+1} (صورة/OCR) ---\n{ocr_text}\n\n"
            
            self.textbox.insert("0.0", full_text)
            self.status_label.configure(text="Done", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"خطأ: {str(e)}", text_color="red")

if __name__ == "__main__":
    app = PDFReaderApp()
    app.mainloop()