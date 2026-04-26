from __future__ import annotations

import logging
import os
import re
from typing import Any

from flask import jsonify
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

TRIGGER_KEEP = {
    "always",
    "never",
    "everyone",
    "nobody",
    "nothing",
    "everything",
    "feel",
    "feels",
    "because",
    "fault",
    "judge",
    "hate",
    "perfect",
    "failure",
    "worst",
    "ruined",
}


class ValidationError(ValueError):
    """Raised when client payload validation fails."""


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
        return

    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))


def get_secret_key() -> str:
    return os.getenv("CB_APP_SECRET", "ai-cognitive-bias-detection-platform")


def api_success(data: Any, *, message: str = "OK", status: int = 200):
    return jsonify({"success": True, "message": message, "data": data}), status


def api_error(
    message: str,
    *,
    status: int = 400,
    code: str = "bad_request",
    details: dict[str, Any] | None = None,
):
    payload = {"success": False, "error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def simple_lemmatize(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess_text(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    processed_tokens: list[str] = []
    for token in tokens:
        if token in ENGLISH_STOP_WORDS and token not in TRIGGER_KEEP:
            continue
        lemma = simple_lemmatize(token)
        if lemma and (lemma not in ENGLISH_STOP_WORDS or lemma in TRIGGER_KEEP):
            processed_tokens.append(lemma)
    return " ".join(processed_tokens)


def validate_text_payload(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        raise ValidationError("Text input is required.")
    if len(value) < 6:
        raise ValidationError("Please enter at least 6 characters for analysis.")
    if len(value) > 1200:
        raise ValidationError("Please keep the input under 1200 characters.")
    return value
