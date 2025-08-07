from src.server.main import db,main_app
from src.config.database import generateDatabase

class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True)
    name = db.Column(db.String(255))
    password = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())
    
    def __init__(self, email, name, password):
        self.name = name
        self.email = email
        self.password = password
        self.created_at = db.func.now()
        self.updated_at = db.func.now()
