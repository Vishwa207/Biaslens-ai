from datetime import datetime
from .db import db

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.String(255), default=lambda: datetime.utcnow().isoformat(timespec="seconds"))

class AnalysisItem(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    bias = db.Column(db.String(255), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    confidence_percent = db.Column(db.String(50), nullable=False)
    thinking_score = db.Column(db.Integer, nullable=False, default=0)
    thinking_band = db.Column(db.String(100), nullable=False, default='Monitor')
    explanation = db.Column(db.Text, nullable=False)
    balanced_thought = db.Column(db.Text, nullable=False, default='')
    suggestion = db.Column(db.Text, nullable=False)
    matched_terms = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.String(255), default=lambda: datetime.utcnow().isoformat(timespec="seconds"))

class SavedItem(db.Model):
    __tablename__ = 'saved_items'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.String(255), default=lambda: datetime.utcnow().isoformat(timespec="seconds"))

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    bias = db.Column(db.String(255), nullable=False, default='')
    suggestion = db.Column(db.Text, nullable=False, default='')
    confidence_percent = db.Column(db.String(50), nullable=False, default='')
    created_at = db.Column(db.String(255), default=lambda: datetime.utcnow().isoformat(timespec="seconds"))
