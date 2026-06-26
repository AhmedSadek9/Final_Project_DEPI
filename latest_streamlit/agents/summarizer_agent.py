def summarizer_agent(context, call_ollama_func):

    prompt = f"""
Summarize clearly in bullet points.

Context:
{context}
"""

    summary = call_ollama_func(prompt)

    return summary