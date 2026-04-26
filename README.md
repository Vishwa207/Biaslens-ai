# AI Cognitive Bias Detection Platform

Production-style AI web application for detecting cognitive bias in user language, scoring confidence, and generating explainable reasoning through a polished SaaS-style interface.

## What Is Included

- Secure authentication flow
  - signup
  - login
  - logout
  - session-protected dashboard routes
- Modular Flask backend
  - `backend/main.py` app factory and routes
  - `backend/model.py` machine-learning service
  - `backend/auth.py` SQLite-backed auth service
  - `backend/store.py` session-scoped prediction history store
  - `backend/utils.py` validation, preprocessing, and API helpers
- AI/NLP pipeline
  - lowercase normalization
  - stopword handling
  - lightweight lemmatization
  - TF-IDF n-grams
  - calibrated Linear SVM baseline
- Explainability engine with keyword-triggered reasoning
- Premium frontend
  - glassmorphism layout
  - dark/light mode
  - animated loading states
  - typed explanation effect
  - confidence progress bar
  - saved drafts and recent results
  - analytics charts with fallback rendering

## Cognitive Bias Classes

- Overgeneralization
- Catastrophizing
- Black-and-White Thinking
- Emotional Reasoning
- Mind Reading
- Personalization
- Balanced Thinking

## Cleaned Project Structure

```text
project/
├── backend/
│   ├── __init__.py
│   ├── auth.py
│   ├── main.py
│   ├── model.py
│   ├── model_loader.py
│   ├── store.py
│   └── utils.py
├── data/
│   └── cognitive_bias_dataset.json
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── app.js
│   └── templates/
│       ├── about.html
│       ├── auth.html
│       ├── base.html
│       ├── index.html
│       └── insights.html
├── model/
│   └── baseline_metrics.json
├── website/
│   └── app.py
├── requirements.txt
└── README.md
```

## Routes

### Pages

- `GET /` redirects to login or dashboard
- `GET /login`
- `GET /signup`
- `POST /logout`
- `GET /dashboard`
- `GET /insights`
- `GET /about`

### APIs

- `POST /predict`
- `POST /save`
- `GET /analytics`
- `GET /health`

## API Response Shape

Example `POST /predict` response:

```json
{
  "success": true,
  "message": "Prediction completed successfully.",
  "data": {
    "prediction": {
      "bias": "Overgeneralization",
      "confidence": 0.92,
      "confidence_percent": "92.0%",
      "explanation": "Absolute language like 'always' suggests a single event is being generalized across every situation."
    }
  }
}
```

## Run Locally

```bash
pip install -r requirements.txt
python website/app.py
```

Open `http://127.0.0.1:5000`.

## Notes

- The auth database is created automatically at `data/app_auth.db` on first run.
- Model evaluation metrics are written to `model/baseline_metrics.json`.
- Transformer support is optional and activates only when dependencies are available.
