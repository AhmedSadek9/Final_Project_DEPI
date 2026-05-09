
import streamlit as st
import os
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import requests
import re

st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title(" AI Study Assistant")


st.sidebar.header(" Controls")
#USER PROFILE DISPLAY
profile = st.session_state.user_profile

st.sidebar.subheader("User Profile")
st.sidebar.write(f"Level: {profile['level']:.2f}")

if profile["weak_topics"]:
    st.sidebar.write("Weak Topics:")
    for topic, score in profile["weak_topics"].items():
        st.sidebar.write(f"- {topic[:30]} ({score:.2f})")
mode = st.sidebar.radio("Mode", ["Normal", "Study"])

uploaded_files = st.sidebar.file_uploader(
    " Upload PDFs", type="pdf", accept_multiple_files=True
)


# SESSION STATE

for key in [
    "answer","doc","context","question",
    "mcqs_raw","written_raw",
    "summary","learning","compare",
    "last_chunk_ids"  
]:
    
    # USER PROFILE
    if "user_profile" not in st.session_state:
       st.session_state.user_profile = {
          "level": 0.5,
          "weak_topics": {},
          "history": []
    }
    
    if "chunk_feedback" not in st.session_state:
        st.session_state.chunk_feedback = {}
        
    if key not in st.session_state:
        st.session_state[key] = None


# FEEDBACK

def give_feedback(chunk_ids, liked=True):
    for cid in chunk_ids:
        if cid not in st.session_state.chunk_feedback:
            st.session_state.chunk_feedback[cid] = 0
        st.session_state.chunk_feedback[cid] += 0.2 if liked else -0.2


def call_ollama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        ).json()
        return res.get("response", "")
    except:
        return "Ollama not running"
    
    

# CHAPTER SPLIT

def split_into_chapters(text):
    import re

    patterns = [
        r"(Chapter\s+\d+.*)",
        r"(\d+\.\s+.+)",
        r"([A-Z][A-Z\s]{5,})",
    ]

    combined = "|".join(patterns)
    splits = re.split(combined, text)

    chapters = []
    current = ""

    for part in splits:
        if not part:
            continue

        if re.match(combined, part.strip()):
            if current:
                chapters.append(current.strip())
            current = part
        else:
            current += " " + part

    if current:
        chapters.append(current.strip())

    return chapters


#BUILD INDEX

@st.cache_resource
def build_index(files):
    chunks, sources, chunk_to_chapter, chunk_ids = [], [], [], []

    for file in files:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

            chapters = split_into_chapters(text)

            for chapter in chapters:

              ch_chunks = [chapter[i:i+800] for i in range(0, len(chapter), 700)]

              for ch in ch_chunks:
                  chunks.append(ch)
                  chunk_ids.append(len(chunks))
                  chunk_to_chapter.append(chapter)
                  sources.append(file.name)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    return index, embed_model, chunks, sources, chunk_to_chapter, chunk_ids

if not uploaded_files:
    st.warning("⬆ Upload PDFs from sidebar")
    st.stop()

index, embed_model, chunks, sources, chunk_to_chapter, chunk_ids = build_index(uploaded_files)
full_text = " ".join(chunks)
st.session_state.all_chapters = split_into_chapters(full_text)


# CHAPTER DETECTION

def detect_best_chapter_from_chunks(question):

    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), k=10)

    chapter_scores = {}

    for idx in I[0]:
        chapter = chunk_to_chapter[idx]

        if chapter not in chapter_scores:
            chapter_scores[chapter] = 0

        chapter_scores[chapter] += 1   

    best_chapter = max(chapter_scores, key=chapter_scores.get)

    return best_chapter

#CORE

       
def get_context(question, selected_chapter=None):

    if selected_chapter:
        best_chapter = selected_chapter
    else:
        best_chapter = detect_best_chapter_from_chunks(question)

    chapter_chunks = [
        chunks[i] for i in range(len(chunks))
        if chunk_to_chapter[i] == best_chapter
    ]

    
    if len(chapter_chunks) == 0:
      
        chapter_chunks = chunks

    
    ch_embeddings = embed_model.encode(chapter_chunks)

    temp_index = faiss.IndexFlatL2(ch_embeddings.shape[1])
    temp_index.add(np.array(ch_embeddings).astype('float32'))

    q_embed = embed_model.encode([question])[0]

    scores = []

    chapter_indices = [
     i for i in range(len(chunks))
       if chunk_to_chapter[i] == best_chapter]
    for i, emb in zip(chapter_indices, ch_embeddings):
      sim = np.dot(q_embed, emb)

      feedback = st.session_state.chunk_feedback.get(i, 0)

      profile = st.session_state.user_profile

      level = profile["level"]
      weak_topics = profile["weak_topics"]
      chapter_weight = 0
      
      if best_chapter in weak_topics:
         chapter_weight = weak_topics[best_chapter]
         
      final_score = sim + feedback + (0.3 * chapter_weight) + (0.2 * (1 - level))   



      scores.append((final_score, i))

    scores.sort(reverse=True)

    top_ids = [i for _, i in scores[:3]]

    st.session_state.last_chunk_ids = top_ids

    selected = [chunks[i] for i in top_ids]
    

    context = "\n\n".join(selected)

    return context, best_chapter[:100]

def compute_confidence(question):
    q_embed = embed_model.encode([question])
    D, I = index.search(np.array(q_embed).astype('float32'), k=3)

    selected = [chunks[i] for i in I[0]]

    context = "\n\n".join(selected)

    avg_dist = sum(D[0]) / len(D[0])
    similarity_score = 1 / (1 + avg_dist)

    chunk_embeds = embed_model.encode([chunks[i] for i in I[0]])
    sim_matrix = np.dot(chunk_embeds, chunk_embeds.T)

    agreement_score = np.mean(sim_matrix)
    max_val = np.max(sim_matrix) if np.max(sim_matrix) != 0 else 1

    confidence = (0.6 * similarity_score) + (0.4 * (agreement_score / max_val))
    return float(max(0.0, min(confidence, 1.0)))



def is_ambiguous_question(question):
    question = question.strip().lower()


    if len(question.split()) < 3:
        return True

  
    vague_words = ["explain", "tell me", "what is this", "info", "details"]
    if any(word in question for word in vague_words):
        return True

    if len(question) < 10:
        return True

    return False


def recommend_topic():
    profile = st.session_state.user_profile

    if not profile["weak_topics"]:
        return None

    weakest = max(profile["weak_topics"], key=profile["weak_topics"].get)
    return weakest



def ask_ai(question, selected_chapter=None):
    context, docs = get_context(question, selected_chapter)

    prompt = f"""
Answer ONLY using context.

Context:
{context}

Question: {question}
Answer:
"""
    return call_ollama(prompt), docs, context



# CHAPTER SELECTOR

selected_chapter = None

if "all_chapters" in st.session_state and st.session_state.all_chapters:

    chapter_titles = [ch.split("\n")[0][:60] for ch in st.session_state.all_chapters]

    selected_title = st.selectbox(
        " Choose Chapter (optional)",
        ["Auto Detect"] + chapter_titles
    )

    if selected_title != "Auto Detect":
        idx = chapter_titles.index(selected_title)
        selected_chapter = st.session_state.all_chapters[idx]


#ASK

question = st.text_input(" Ask your question")

if st.button("Ask"):

    if is_ambiguous_question(question):
        st.warning(" Your question is unclear. Try specifying topic, chapter, or keywords")
    else:
        ans, doc, ctx = ask_ai(question, selected_chapter)
        st.session_state.answer = ans
        st.session_state.doc = doc
        st.session_state.context = ctx
        st.session_state.question = question


#  OUTPUT

if st.session_state.answer:
    rec = recommend_topic()
    if rec:
      st.info(f" You should review: {rec[:50]}")
    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader(" Source")
    st.write(st.session_state.doc)

    conf = compute_confidence(st.session_state.question)
    st.progress(conf)
    st.write(f"{conf*100:.1f}% confidence")

    col1, col2 = st.columns(2)
    if col1.button("Like"):
        give_feedback(st.session_state.last_chunk_ids, True)
    if col2.button("Dislike"):
        give_feedback(st.session_state.last_chunk_ids, False)


#STUDY MODE

if mode == "Study" and st.session_state.answer:

    tabs = st.tabs([" Summary"," Learning"," MCQ"," Written"," Compare"])


    #SUMMARY

    with tabs[0]:

        if st.button("Generate Summary"):
            st.session_state.summary = call_ollama(f"""
Summarize clearly in bullet points.

Context:
{st.session_state.context}
""")

        if st.session_state.summary:
            st.write(st.session_state.summary)


    # LEARNING

    with tabs[1]:

        if st.button("Learning Path"):
            st.session_state.learning = call_ollama(f"""
Create structured learning path in bullets.

Context:
{st.session_state.context}
""")

        if st.session_state.learning:
            st.write(st.session_state.learning)


    #  MCQ
  
    with tabs[2]:
        num_questions = st.slider("Number of Questions", 1, 10, 3)

        difficulty = st.selectbox(
             "Difficulty Level",
             ["Easy", "Medium", "Hard"]
    )


        if st.button("Generate MCQ", key="gen_mcq") or st.session_state.mcqs_raw is None:
            if difficulty == "Easy":
               instruction = "Focus on direct facts and definitions."
            elif difficulty == "Medium":
               instruction = "Require understanding and explanation."
            else:
               instruction = "Require analysis, tricky reasoning, and inference."
            st.session_state.mcqs_raw = call_ollama(f"""
Generate EXACTLY {num_questions} MCQs.

Difficulty: {difficulty}

Instructions:
{instruction}

STRICT FORMAT:
Q1: ...
A) ...
B) ...
C) ...
D) ...
Answer: A

IMPORTANT:
- You MUST generate EXACTLY {num_questions} questions.
- Do NOT add extra text.
- Do NOT break format.

Context:
{st.session_state.context}
""")

        if st.session_state.mcqs_raw:

            blocks = re.split(r"Q\d:", st.session_state.mcqs_raw)[1:]

            user_answers = []

            for i, block in enumerate(blocks):

                lines = block.strip().split("\n")
                question_text = lines[0]

                options = [l for l in lines if re.match(r"[A-D][\)\.]", l)]
                answer_line = [l for l in lines if "Answer:" in l]

                if answer_line:
                     correct = answer_line[0].split(":")[1].strip()
                else:
                     correct = "A"     

                st.write(f"**Q{i+1}: {question_text}**")

                choice = st.radio(
                    f"Select answer Q{i+1}",
                    options,
                    key=f"mcq_{i}"
                )

                user_answers.append((choice[0], correct))

            if st.button("Check Answers"):

               correct_count = 0

               for i, (u, c) in enumerate(user_answers):
                      if u == c:
                         st.success(f"Q{i+1}: Correct ")
                         correct_count += 1
                      else:
                         st.error(f"Q{i+1}: Wrong  (Correct: {c})")


            accuracy = correct_count / len(user_answers)

            profile = st.session_state.user_profile


            profile["level"] = (profile["level"] + accuracy) / 2

  
            current_chapter = st.session_state.doc
            if current_chapter:
                 profile["weak_topics"][current_chapter] = 1 - accuracy

    
            profile["history"].append({
        "question": st.session_state.question,
        "accuracy": accuracy,
        "chapter": current_chapter
    })

  
    #WRITTEN
  
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

  
    #COMPARE

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