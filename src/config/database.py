from flask_sqlalchemy import SQLAlchemy
from src.config.config import DATABASE_URL

def database(app):
  global db
  if 'db' not in globals():
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db = SQLAlchemy(app)
    print("[INFO] Database connected")
  return db
def generateDatabase(main_app,db):
  with main_app.app_context():
    db.create_all()