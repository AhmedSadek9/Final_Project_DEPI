def evaluate_answer(question, answer, call_ollama_func):

    prompt = f"""
Grade this strictly.

Question:
{question}

Student Answer:
{answer}

Return ONLY:

Score: X/10

Mistakes:
- ...

Better answer:
- ...
"""

    feedback = call_ollama_func(prompt)

    return feedback