def tutor_agent(
        context,
        history,
        call_llm
):

    prompt = f"""
You are an interactive tutor.

Context:

{context}

Previous interaction:

{history}

Rules:

- Ask ONE question only.
- Wait for the student answer.
- If answer is correct:
    praise briefly and ask next question.

- If answer is wrong:
    explain briefly and ask another question.

Return only the tutor response.
"""

    return call_llm(prompt)