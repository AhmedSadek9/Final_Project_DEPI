def planner_agent(
    context,
    weak_topics,
    level,
    history,
    call_ollama_func
):

    prompt = f"""
Build a personalized learning roadmap.

Student Level:
{level}

Weak Topics:
{weak_topics}

History:
{history}

Context:
{context}

Requirements:

- Order topics from easy to difficult.
- Focus more on weak topics.
- Give prerequisites first.
- Use bullet points.

Output example:

Embedded Systems
↓
Sensors
↓
ADC
↓
DAC
↓
Communication Protocols
↓
RTOS
"""

    return call_ollama_func(prompt)