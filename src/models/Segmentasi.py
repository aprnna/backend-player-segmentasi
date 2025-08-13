from src.server.main import db,main_app
from src.config.database import generateDatabase

class Segmentasi(db.Model):
    __tablename__ = 'segmentasi'
    Segmentasi_id = db.Column(db.Integer, primary_key=True)
    Proses_id = db.Column(db.Integer,db.ForeignKey('steam_id_proses.Proses_id'))   
    segmentation_csv_path = db.Column(db.String(255))
    membership_csv_path = db.Column(db.String(255))
    karakteristik_json_path = db.Column(db.String(255)) 
    interpretasi_json_path = db.Column(db.String(255))
    
    created_at = db.Column(db.DateTime, default=db.func.now())

    def __init__(self, Proses_id, segmentation_csv_path, membership_csv_path, karakteristik_json_path, interpretasi_json_path):
        self.Proses_id = Proses_id
        self.segmentation_csv_path = segmentation_csv_path
        self.membership_csv_path = membership_csv_path
        self.karakteristik_json_path = karakteristik_json_path
        self.interpretasi_json_path = interpretasi_json_path

