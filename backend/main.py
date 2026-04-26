from __future__ import annotations

import logging

from flask import Flask, g, redirect, render_template, request, session, url_for

from .auth import AuthService, api_login_required, page_login_required, safe_redirect_target
from .db import db
from .model import EXAMPLE_BANK
from .model_loader import get_model
from .store import DataStore
from .utils import ValidationError, api_error, api_success, configure_logging, get_secret_key, validate_text_payload
import os

LOGGER = logging.getLogger(__name__)

store = DataStore()
auth_service = AuthService()


def create_app() -> Flask:
    configure_logging()

    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )
    app.secret_key = get_secret_key()
    
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
        auth_service.initialize()
        store.initialize()

    model = get_model()

    @app.before_request
    def prepare_session() -> None:
        g.current_user = auth_service.get_user_by_id(session.get("user_id"))
        if session.get("user_id") and g.current_user is None:
            session.pop("user_id", None)
            session.modified = True

    @app.after_request
    def apply_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    @app.context_processor
    def inject_global_context() -> dict:
        return {
            "current_user": getattr(g, "current_user", None),
            "is_auth_page": request.endpoint in {"login", "signup"},
        }

    def active_user_id() -> int:
        if not g.current_user:
            raise ValidationError("Authentication required.")
        return int(g.current_user["id"])

    def page_context() -> dict:
        user_id = active_user_id()
        history = model.enrich_history(store.get_history(user_id))
        saved_items = store.get_saved_items(user_id)
        chat_messages = store.get_chat_messages(user_id)
        analytics = model.analytics_snapshot(history, saved_items, chat_messages)
        return {
            "metrics": model.metrics,
            "examples": EXAMPLE_BANK,
            "history": history,
            "saved_items": saved_items,
            "chat_messages": chat_messages,
            "analytics": analytics,
            "latest": history[0] if history else None,
        }

    @app.get("/")
    def root():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.current_user:
            return redirect(url_for("dashboard"))

        auth_mode = (request.values.get("auth_mode") or request.args.get("mode") or "login").strip().lower()
        auth_mode = auth_mode if auth_mode in {"login", "signup"} else "login"
        next_target = safe_redirect_target(request.values.get("next") or request.args.get("next")) or url_for("dashboard")

        values = {
            "login_email": "",
            "signup_name": "",
            "signup_email": "",
        }
        error_message = None

        if request.method == "POST":
            if auth_mode == "signup":
                values["signup_name"] = (request.form.get("full_name") or "").strip()
                values["signup_email"] = (request.form.get("email") or "").strip()
                password = request.form.get("password") or ""
                confirm_password = request.form.get("confirm_password") or ""
                try:
                    user = auth_service.create_user(
                        values["signup_name"],
                        values["signup_email"],
                        password,
                        confirm_password,
                    )
                    session["user_id"] = user["id"]
                    session.modified = True
                    return redirect(next_target)
                except ValidationError as error:
                    error_message = str(error)
            else:
                values["login_email"] = (request.form.get("email") or "").strip()
                password = request.form.get("password") or ""
                try:
                    user = auth_service.authenticate(values["login_email"], password)
                    session["user_id"] = user["id"]
                    session.modified = True
                    return redirect(next_target)
                except ValidationError as error:
                    error_message = str(error)

        return render_template(
            "auth.html",
            page="login",
            auth_mode=auth_mode,
            error_message=error_message,
            form_values=values,
            next_target=next_target,
            metrics=model.metrics,
        )

    @app.get("/signup")
    def signup():
        return redirect(url_for("login", mode="signup"))

    @app.post("/logout")
    def logout():
        session.pop("user_id", None)
        session.modified = True
        return redirect(url_for("login"))

    @app.get("/dashboard")
    @page_login_required
    def dashboard():
        return render_template("index.html", page="dashboard", **page_context())

    @app.get("/insights")
    @page_login_required
    def insights():
        context = page_context()
        history = context["history"]
        selected_id = request.args.get("item", type=int)
        selected_item = next((item for item in history if item["id"] == selected_id), history[0] if history else None)
        return render_template(
            "insights.html",
            page="insights",
            metrics=context["metrics"],
            analytics=context["analytics"],
            history=history,
            selected_item=selected_item,
        )

    @app.get("/about")
    @page_login_required
    def about():
        context = page_context()
        return render_template("about.html", page="about", metrics=context["metrics"], analytics=context["analytics"])

    @app.post("/predict")
    @api_login_required
    def predict():
        payload = request.get_json(silent=True) or {}
        text = validate_text_payload(payload.get("text"))
        prediction = model.predict(text)
        user_id = active_user_id()

        history = store.add_history_item(
            user_id,
            {
                "text": text,
                **prediction.to_dict(),
            },
        )
        history = model.enrich_history(history)
        saved_items = store.get_saved_items(user_id)
        chat_messages = store.get_chat_messages(user_id)
        analytics = model.analytics_snapshot(history, saved_items, chat_messages)

        LOGGER.info("Prediction created for user=%s bias=%s confidence=%s", user_id, prediction.bias, prediction.confidence)
        return api_success(
            {
                "prediction": history[0],
                "history": history,
                "saved_items": saved_items,
                "chat_messages": chat_messages,
                "analytics": analytics,
            },
            message="Prediction completed successfully.",
        )

    @app.post("/save")
    @api_login_required
    def save_item():
        payload = request.get_json(silent=True) or {}
        text = validate_text_payload(payload.get("text"))
        user_id = active_user_id()

        saved_items = store.add_saved_item(
            user_id,
            {
                "text": text,
            },
        )
        history = model.enrich_history(store.get_history(user_id))
        chat_messages = store.get_chat_messages(user_id)
        analytics = model.analytics_snapshot(history, saved_items, chat_messages)

        return api_success(
            {
                "saved_items": saved_items,
                "analytics": analytics,
            },
            message="Draft saved successfully.",
        )

    @app.post("/chat")
    @api_login_required
    def chat():
        payload = request.get_json(silent=True) or {}
        text = validate_text_payload(payload.get("text"))
        user_id = active_user_id()
        recent_messages = store.get_chat_messages(user_id, limit=8)
        prediction = model.predict(text)
        assistant_reply = model.build_chat_reply(text, prediction, recent_messages)

        store.add_chat_message(
            user_id,
            {
                "role": "user",
                "content": text,
            },
        )
        messages = store.add_chat_message(
            user_id,
            {
                "role": "assistant",
                "content": assistant_reply.content,
                "bias": assistant_reply.bias,
                "suggestion": assistant_reply.suggestion,
                "confidence_percent": prediction.confidence_percent,
            },
        )

        history = model.enrich_history(store.get_history(user_id))
        saved_items = store.get_saved_items(user_id)
        analytics = model.analytics_snapshot(history, saved_items, messages)

        return api_success(
            {
                "reply": {
                    **prediction.to_dict(),
                    **assistant_reply.to_dict(),
                },
                "messages": messages,
                "analytics": analytics,
            },
            message="Assistant response generated successfully.",
        )

    @app.get("/analytics")
    @api_login_required
    def analytics():
        user_id = active_user_id()
        history = model.enrich_history(store.get_history(user_id))
        saved_items = store.get_saved_items(user_id)
        chat_messages = store.get_chat_messages(user_id)
        return api_success(model.analytics_snapshot(history, saved_items, chat_messages))

    @app.get("/health")
    def health():
        return api_success(
            {
                "status": "ok",
                "model_name": model.metrics.get("model_name", "TF-IDF + Linear SVM"),
                "transformer_ready": model.metrics.get("transformer_ready", False),
            }
        )

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        return api_error(str(error), status=422, code="validation_error")

    @app.errorhandler(404)
    def not_found(_error):
        return api_error("Resource not found.", status=404, code="not_found")

    @app.errorhandler(500)
    def internal_error(error):
        LOGGER.exception("Unhandled error: %s", error)
        return api_error("Internal server error.", status=500, code="internal_error")

    return app
