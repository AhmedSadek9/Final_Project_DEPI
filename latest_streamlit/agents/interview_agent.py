def interview_agent(
        context,
        history,
        call_llm
):

    prompt = f"""
You are a technical interviewer.

Context:

{context}

Interview History:

{history}

Rules:

- Ask ONE interview question only.
- Wait for the student answer.
- Give score out of 10.
- Mention strengths and mistakes briefly.
- Ask next question.

Return only the interviewer response.
"""

    return call_llm(prompt)