import os
from dotenv import load_dotenv
load_dotenv()

from backend.main import create_app
from backend.db import db
from backend.models_db import User

app = create_app()

with app.app_context():
    print("Connecting to Supabase PostgreSQL...")
    db.create_all()
    print("Tables created successfully!")
    users = User.query.all()
    print(f"Total users in DB: {len(users)}")
