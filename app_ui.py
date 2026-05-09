import streamlit as st
import os
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import requests
import re

st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("📚 AI Study Assistant")

st.sidebar.header(" Controls")
mode = st.sidebar.radio("Mode", ["Normal", "Study"])

uploaded_files = st.sidebar.file_uploader(
    " Upload PDFs", type="pdf", accept_multiple_files=True
)

for key in [
    "answer","doc","context","question",
    "mcqs_raw","written_raw",
    "summary","learning","compare"
]:
    if key not in st.session_state:
        st.session_state[key] = None

feedback_scores = {}

def give_feedback(doc, liked=True):
    docs = doc.split(", ")
    for d in docs:
        if d not in feedback_scores:
            feedback_scores[d] = 0
        feedback_scores[d] += 0.2 if liked else -0.2

def call_ollama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        ).json()
        return res.get("response", "")
    except:
        return "Ollama not running"

@st.cache_resource
def build_index(files):
    chunks, sources = [], []

    for file in files:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

            file_chunks = [text[i:i+800] for i in range(0, len(text), 700)]
            for ch in file_chunks:
                chunks.append(ch)
                sources.append(file.name)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    return index, embed_model, chunks, sources

if not uploaded_files:
    st.warning("⬆ Upload PDFs from sidebar")
    st.stop()

index, embed_model, chunks, sources = build_index(uploaded_files)

def get_context(question):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), k=8)

    selected_chunks = []
    selected_docs = set()

    for idx in I[0]:
        selected_chunks.append(chunks[idx])
        selected_docs.add(sources[idx])

    context = "\n\n---\n\n".join(selected_chunks[:4])
    docs = ", ".join(selected_docs)

    return context, docs

def compute_confidence(question):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), k=5)

    avg_dist = sum(D[0]) / len(D[0])
    similarity_score = 1 / (1 + avg_dist)

    chunk_embeds = embed_model.encode([chunks[i] for i in I[0]])
    sim_matrix = np.dot(chunk_embeds, chunk_embeds.T)

    agreement_score = np.mean(sim_matrix)
    max_val = np.max(sim_matrix) if np.max(sim_matrix) != 0 else 1

    confidence = (0.6 * similarity_score) + (0.4 * (agreement_score / max_val))
    return float(max(0.0, min(confidence, 1.0)))

def ask_ai(question):
    context, docs = get_context(question)

    prompt = f"""
Answer ONLY using context.

Context:
{context}

Question: {question}
Answer:
"""
    return call_ollama(prompt), docs, context

question = st.text_input(" Ask your question")

if st.button("Ask"):
    ans, doc, ctx = ask_ai(question)
    st.session_state.answer = ans
    st.session_state.doc = doc
    st.session_state.context = ctx
    st.session_state.question = question

if st.session_state.answer:

    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader(" Source")
    st.write(st.session_state.doc)

    conf = compute_confidence(st.session_state.question)
    st.progress(conf)
    st.write(f"{conf*100:.1f}% confidence")

    col1, col2 = st.columns(2)
    if col1.button("Like"):
        give_feedback(st.session_state.doc, True)
    if col2.button("Dislike"):
        give_feedback(st.session_state.doc, False)

if mode == "Study" and st.session_state.answer:

    tabs = st.tabs([" Summary"," Learning"," MCQ"," Written"," Compare"])

    with tabs[0]:

        if st.button("Generate Summary"):
            st.session_state.summary = call_ollama(f"""
Summarize clearly in bullet points.

Context:
{st.session_state.context}
""")

        if st.session_state.summary:
            st.write(st.session_state.summary)

    with tabs[1]:

        if st.button("Learning Path"):
            st.session_state.learning = call_ollama(f"""
Create structured learning path in bullets.

Context:
{st.session_state.context}
""")

        if st.session_state.learning:
            st.write(st.session_state.learning)

    with tabs[2]:

        if st.button("Generate MCQ"):
            st.session_state.mcqs_raw = call_ollama(f"""
Generate 3 MCQs.

STRICT FORMAT:
Q1: ...
A) ...
B) ...
C) ...
D) ...
Answer: A

Context:
{st.session_state.context}
""")

        if st.session_state.mcqs_raw:

            blocks = re.split(r"Q\d:", st.session_state.mcqs_raw)[1:]

            user_answers = []

            for i, block in enumerate(blocks):

                lines = block.strip().split("\n")
                question_text = lines[0]

                options = [l for l in lines if re.match(r"[A-D]\)", l)]
                answer_line = [l for l in lines if "Answer:" in l]

                correct = answer_line[0].split(":")[1].strip() if answer_line else "A"

                st.write(f"**Q{i+1}: {question_text}**")

                choice = st.radio(
                    f"Select answer Q{i+1}",
                    options,
                    key=f"mcq_{i}"
                )

                user_answers.append((choice[0], correct))

            if st.button("Check Answers"):
                for i, (u, c) in enumerate(user_answers):
                    if u == c:
                        st.success(f"Q{i+1}: Correct ")
                    else:
                        st.error(f"Q{i+1}: Wrong  (Correct: {c})")

    with tabs[3]:

        if st.button("Generate Questions"):
            st.session_state.written_raw = call_ollama(f"""
Generate EXACTLY 2 SHORT questions.

FORMAT:
Q1: ...
Q2: ...

Context:
{st.session_state.context}
""")

        if st.session_state.written_raw:

            questions = re.findall(r"Q\d:\s*(.*)", st.session_state.written_raw)

            answers = []

            for i, q in enumerate(questions):
                st.write(f"**Q{i+1}: {q}**")

                ans = st.text_area(
                    f"Answer Q{i+1}",
                    key=f"written_{i}"
                )

                answers.append((q, ans))

            if st.button("Evaluate Answers"):

                for i, (q, ans) in enumerate(answers):

                    st.markdown(f"### Feedback Q{i+1}")

                    if not ans.strip():
                        st.error("No answer (0/10)")
                    else:
                        feedback = call_ollama(f"""
Grade this strictly.

Question: {q}
Answer: {ans}

Return ONLY:

Score: X/10
Mistakes:
- ...
Better answer:
- ...
""")
                        st.write(feedback)

    with tabs[4]:

        if st.button("Compare"):
            st.session_state.compare = call_ollama(f"""
Compare clearly.

Use:
- Similarities
- Differences

Context:
{st.session_state.context}
""")

        if st.session_state.compare:
            st.write(st.session_state.compare)