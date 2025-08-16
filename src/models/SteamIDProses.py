from src.server.main import db,main_app ,  proses_topic_association
from src.config.database import generateDatabase

class SteamIDProses(db.Model):
    __tablename__ = 'steam_id_proses'
    Proses_id = db.Column(db.Integer, primary_key=True)
    Steam_id = db.Column(db.Text)
    User_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    
    segmentasi_entries = db.relationship('Segmentasi', backref='proses', lazy=True)

    topics = db.relationship(
        'TopicModeling', 
        secondary=proses_topic_association, 
        back_populates='steam_proses_entries',
    )

    def __init__(self, Steam_ids, User_id):
        self.Steam_id = Steam_ids
        self.User_id = User_id
