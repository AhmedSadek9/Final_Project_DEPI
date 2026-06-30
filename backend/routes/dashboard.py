from fastapi import APIRouter

from backend.data.mongodb import (
    documents_collection,
    history_collection,
    progress_collection
)

router = APIRouter()


@router.get("/dashboard")
def dashboard():

    documents = documents_collection.count_documents({})

    history = history_collection.count_documents({})

    progress = progress_collection.find_one()

    accuracy = 0
    study_hours = 0
    weak_topics = []

    if progress:

        accuracy = progress.get(
            "accuracy",
            0
        )

        study_hours = progress.get(
            "study_hours",
            0
        )

        weak_topics = progress.get(
            "weak_topics",
            []
        )

    return {

        "documents": documents,

        "history": history,

        "accuracy": accuracy,

        "study_hours": study_hours,

        "weak_topics": weak_topics

    }