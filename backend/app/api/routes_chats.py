"""Chat history: list past chats, load one's full entry history, delete one. Creating a chat and
adding entries to it both happen as a side effect of POST /api/query (see routes_query.py) -- there
is no separate "create chat" endpoint, since a chat with zero entries isn't a useful thing to have
sitting in the list."""

from fastapi import APIRouter, HTTPException

from app import chat_store
from app.schemas.models import ChatDetailOut, ChatSummaryOut

router = APIRouter()


@router.get("/chats", response_model=list[ChatSummaryOut])
async def list_chats() -> list[ChatSummaryOut]:
    return [ChatSummaryOut(**c) for c in chat_store.list_chats()]


@router.get("/chats/{chat_id}", response_model=ChatDetailOut)
async def get_chat(chat_id: str) -> ChatDetailOut:
    chat = chat_store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail=f"Unknown chat_id '{chat_id}'.")
    return ChatDetailOut(**chat)


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, bool]:
    deleted = chat_store.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown chat_id '{chat_id}'.")
    return {"deleted": True}
