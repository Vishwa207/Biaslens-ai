from __future__ import annotations

from datetime import datetime
import json
from threading import Lock
from typing import Any

from .db import db
from .models_db import AnalysisItem, SavedItem, ChatMessage
from .utils import normalize_text


class DataStore:
    """SQLAlchemy-backed store for analysis history, saved drafts, and chat."""

    def __init__(self) -> None:
        self._lock = Lock()

    def initialize(self) -> None:
        with db.engine.connect() as conn:
            # We don't need to manually create tables if Flask-Migrate or db.create_all() is used.
            # But just in case, we will rely on main.py to call db.create_all()
            pass

    def add_history_item(self, user_id: int, item: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            matched_terms = json.dumps(item.get("matched_terms", []))
            new_item = AnalysisItem(
                user_id=user_id,
                text=item["text"],
                bias=item["bias"],
                confidence=item["confidence"],
                confidence_percent=item["confidence_percent"],
                thinking_score=item.get("thinking_score", 0),
                thinking_band=item.get("thinking_band", "Monitor"),
                explanation=item["explanation"],
                balanced_thought=item.get("balanced_thought", ""),
                suggestion=item["suggestion"],
                matched_terms=matched_terms,
            )
            db.session.add(new_item)
            db.session.commit()
            return self.get_history(user_id)

    def get_history(self, user_id: int, limit: int = 150) -> list[dict[str, Any]]:
        items = AnalysisItem.query.filter_by(user_id=user_id).order_by(AnalysisItem.id.desc()).limit(limit).all()
        return [self._serialize_history(item) for item in items]

    def add_saved_item(self, user_id: int, item: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            new_item = SavedItem(
                user_id=user_id,
                text=item["text"],
            )
            db.session.add(new_item)
            db.session.commit()
            return self.get_saved_items(user_id)

    def get_saved_items(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        items = SavedItem.query.filter_by(user_id=user_id).order_by(SavedItem.id.desc()).limit(limit).all()
        return [self._serialize_saved(item) for item in items]

    def add_chat_message(self, user_id: int, message: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            new_msg = ChatMessage(
                user_id=user_id,
                role=message["role"],
                content=message["content"],
                bias=message.get("bias", ""),
                suggestion=message.get("suggestion", ""),
                confidence_percent=message.get("confidence_percent", ""),
            )
            db.session.add(new_msg)
            db.session.commit()
            return self.get_chat_messages(user_id)

    def get_chat_messages(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        items = ChatMessage.query.filter_by(user_id=user_id).order_by(ChatMessage.id.desc()).limit(limit).all()
        return [self._serialize_chat(item) for item in reversed(items)]

    def _serialize_history(self, item: AnalysisItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "text": item.text,
            "bias": item.bias,
            "confidence": item.confidence,
            "confidence_percent": item.confidence_percent,
            "thinking_score": item.thinking_score,
            "thinking_band": item.thinking_band,
            "explanation": item.explanation,
            "balanced_thought": item.balanced_thought,
            "suggestion": item.suggestion,
            "matched_terms": json.loads(item.matched_terms),
            "created_at": item.created_at,
            "sentiment_bucket": "Balanced Thinking" if item.bias == "Balanced Thinking" else "Negative Thinking"
        }

    def _serialize_saved(self, item: SavedItem) -> dict[str, Any]:
        return {
            "id": item.id,
            "text": item.text,
            "created_at": item.created_at,
        }

    def _serialize_chat(self, item: ChatMessage) -> dict[str, Any]:
        return {
            "id": item.id,
            "role": item.role,
            "content": item.content,
            "bias": item.bias,
            "suggestion": item.suggestion,
            "confidence_percent": item.confidence_percent,
            "created_at": item.created_at,
        }
