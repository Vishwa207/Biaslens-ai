from __future__ import annotations

from datetime import datetime
from functools import wraps
import re
from typing import Any, Callable

from flask import redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.exc import IntegrityError

from .db import db
from .models_db import User
from .utils import ValidationError, api_error

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

class AuthService:
    def __init__(self) -> None:
        pass

    def initialize(self) -> None:
        pass

    def create_user(self, full_name: str, email: str, password: str, confirm_password: str) -> dict[str, Any]:
        normalized_name = full_name.strip()
        normalized_email = self._normalize_email(email)
        self._validate_signup(normalized_name, normalized_email, password, confirm_password)

        new_user = User(
            full_name=normalized_name,
            email=normalized_email,
            password_hash=generate_password_hash(password),
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            raise ValidationError("An account with this email already exists.") from error

        return self.get_user_by_id(new_user.id)

    def authenticate(self, email: str, password: str) -> dict[str, Any]:
        normalized_email = self._normalize_email(email)
        if not normalized_email or not password:
            raise ValidationError("Email and password are required.")

        user = User.query.filter_by(email=normalized_email).first()

        if not user or not check_password_hash(user.password_hash, password):
            raise ValidationError("Invalid email or password.")

        return self._serialize_user(user)

    def get_user_by_id(self, user_id: int | None) -> dict[str, Any] | None:
        if not user_id:
            return None

        user = db.session.get(User, user_id)
        return self._serialize_user(user) if user else None

    def _validate_signup(self, full_name: str, email: str, password: str, confirm_password: str) -> None:
        if len(full_name) < 2:
            raise ValidationError("Please enter your full name.")
        if not EMAIL_PATTERN.match(email):
            raise ValidationError("Please enter a valid email address.")
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long.")
        if password != confirm_password:
            raise ValidationError("Passwords do not match.")

    def _normalize_email(self, email: str) -> str:
        return email.strip().lower()

    def _serialize_user(self, user: User) -> dict[str, Any]:
        return {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "created_at": user.created_at,
        }

def api_login_required(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return api_error("Authentication required.", status=401, code="unauthorized")
        return func(*args, **kwargs)
    return wrapper

def page_login_required(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)
    return wrapper

def safe_redirect_target(target: str | None) -> str | None:
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    return target
