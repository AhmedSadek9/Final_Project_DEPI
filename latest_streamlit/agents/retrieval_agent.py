def retrieval_agent(question, selected_chapter, get_context_func):

    context, docs = get_context_func(
        question,
        selected_chapter
    )

    return {
        "context": context,
        "docs": docs
    }