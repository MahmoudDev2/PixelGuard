# This script initializes the database.
# Run this once before starting the main application for the first time.
from app import app, db

print("Initializing database...")
with app.app_context():
    db.create_all()
print("Database initialized successfully. You can now run the application.")
