import os
import gradio as gr
import pdfplumber
from langchain_groq import ChatGroq

# =========================
# API KEY
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it in Hugging Face Secrets.")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(file_path):
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

    if not text.strip():
        return "⚠️ No readable text found (maybe scanned PDF)."

    return text


# =========================
# SUMMARIZATION (FIXED FOR LONG PDFS)
# =========================
def summarize_text(text, length, style):

    # chunking (IMPORTANT FIX for 70 slides)
    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]

    partial_summaries = []

    for chunk in chunks:
        prompt = f"""
        Summarize this part in {style} style.
        Keep it {length}.
        Text:
        {chunk}
        """
        res = llm.invoke(prompt)
        partial_summaries.append(res.content)

    final_prompt = f"""
    Combine these summaries into one clean final summary:
    {partial_summaries}
    """

    final_res = llm.invoke(final_prompt)
    return final_res.content


# =========================
# MAIN FUNCTION
# =========================
def process_pdf(file, length, style):

    if file is None:
        return "⚠️ Please upload a PDF file."

    file_path = file.name if hasattr(file, "name") else file

    text = extract_text_from_pdf(file_path)

    if "Error" in text or "No readable" in text:
        return text

    return summarize_text(text, length, style)


# =========================
# GRADIO UI
# =========================
demo = gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="📄 Upload PDF"),
        gr.Radio(["Short", "Medium", "Long"], value="Medium", label="Length"),
        gr.Radio(["Bullets", "Key Takeaways", "Paragraph"], value="Key Takeaways", label="Style")
    ],
    outputs="text",
    title="📄 AI PDF Summarizer",
    description="Upload PDF → get smart AI summary (works with long PDFs too)"
)

demo.launch()