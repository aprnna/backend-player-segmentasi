from src.server.main import db,main_app
from src.config.database import generateDatabase

class Segmentasi(db.Model):
    __tablename__ = 'segmentasi'
    Topic_id = db.Column(db.Integer, primary_key=True)
    Proses_id = db.Column(db.Integer,db.ForeignKey('steam_id_proses.Proses_id'))   
    Arketipe = db.Column(db.String(255))

    def __init__(self, Arketipe):
        self.Arketipe = Arketipe
