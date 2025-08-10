from src.server.main import db,main_app
from src.config.database import generateDatabase

class Segmentasi(db.Model):
    __tablename__ = 'segmentasi'
    Segmentasi_id = db.Column(db.Integer, primary_key=True)
    Proses_id = db.Column(db.Integer,db.ForeignKey('steam_id_proses.Proses_id'))   
    steam_id = db.Column(db.String(20), nullable=False, index=True)
    
    # Kolom dari hasil_profil_pemain.csv
    total_games = db.Column(db.Integer)
    avg_playtime = db.Column(db.Float)
    total_achievements = db.Column(db.Integer)
    top_3_genres = db.Column(db.String(255)) 
    dominant_topic_user = db.Column(db.Integer)
    
    # Kolom dari hasil_segmentasi_pengguna.csv
    dominant_archetype = db.Column(db.Integer, index=True)
    archetype_1_weight = db.Column(db.Float)
    archetype_2_weight = db.Column(db.Float)
    archetype_3_weight = db.Column(db.Float)
    archetype_4_weight = db.Column(db.Float)
    archetype_5_weight = db.Column(db.Float)
    archetype_6_weight = db.Column(db.Float)
    archetype_7_weight = db.Column(db.Float)
    archetype_8_weight = db.Column(db.Float)
    archetype_9_weight = db.Column(db.Float)
    archetype_10_weight = db.Column(db.Float)

    def __init__(self, proses_id, steam_id, total_games, avg_playtime, total_achievements, top_3_genres, dominant_topic_user, dominant_archetype, archetype_weights):
        self.Proses_id = proses_id
        self.steam_id = steam_id
        self.total_games = total_games
        self.avg_playtime = avg_playtime
        self.total_achievements = total_achievements
        self.top_3_genres = top_3_genres
        self.dominant_topic_user = dominant_topic_user
        self.dominant_archetype = dominant_archetype
        for i in range(1, 11):
            weight_key = f'archetype_{i}'
            setattr(self, f'{weight_key}_weight', archetype_weights.get(weight_key, 0.0))

