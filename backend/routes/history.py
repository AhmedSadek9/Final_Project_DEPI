from fastapi import APIRouter
from backend.data.mongodb import history_collection

router = APIRouter()


@router.get("/history")
def get_history():

    history = list(
        history_collection.find(
            {},
            {
                "_id": 0
            }
        ).sort("time", -1).limit(10)
    )

    return {
        "history": history
    }