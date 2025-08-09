from src.server.main import db,main_app, proses_topic_association
from src.config.database import generateDatabase

class TopicModeling(db.Model):
    __tablename__ = 'topic_modeling'
    Topic_id = db.Column(db.Integer, primary_key=True)
    Keyword = db.Column(db.Text)
    Cluster = db.Column(db.Integer)    

    steam_proses_entries = db.relationship(
        'SteamIDProses', 
        secondary=proses_topic_association, 
        back_populates='topics',
    )

    def __init__(self, Keyword, Cluster):
        self.Keyword = Keyword
        self.Cluster = Cluster
