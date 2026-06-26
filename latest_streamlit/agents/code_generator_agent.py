def code_generator_agent(
        topic,
        language,
        context,
        call_llm
):

    prompt = f"""
You are an expert programmer.

Topic:

{topic}

Programming Language:

{language}

Context:

{context}

Return:

1. Full code.

2. Step-by-step explanation.

3. Time Complexity.

4. Space Complexity.

5. Best Practices.

6. Common Mistakes.

Format clearly.
"""

    return call_llm(prompt)