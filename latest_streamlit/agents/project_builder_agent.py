def project_builder_agent(
        topic,
        context,
        call_llm
):

    prompt = f"""
You are an expert project supervisor.

Topic:

{topic}

Context:

{context}

Build a complete project.

Return:

1. Project Title

2. Objective

3. Components Needed

4. Hardware Architecture

5. Software Flow

6. File Structure

7. Main Algorithms

8. Testing Plan

9. Report Chapters

10. Presentation Outline

Return clear sections only.
"""

    return call_llm(prompt)