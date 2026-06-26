def generate_mcq(context,
                 num_questions,
                 difficulty,
                 call_ollama_func):

    if difficulty == "Easy":
        instruction = "Focus on direct facts and definitions."

    elif difficulty == "Medium":
        instruction = "Require understanding and explanation."

    else:
        instruction = "Require analysis, tricky reasoning, and inference."

    prompt = f"""
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

Context:
{context}
"""

    return call_ollama_func(prompt)


def generate_written_questions(context,
                               call_ollama_func):

    prompt = f"""
Generate EXACTLY 2 SHORT questions.

FORMAT:

Q1: ...
Q2: ...

Context:
{context}
"""

    return call_ollama_func(prompt)