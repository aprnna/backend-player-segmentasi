from src.config.database import db

class PlayerGameData(db.Model):
    __tablename__ = 'player_game_data'

    id = db.Column(db.Integer, primary_key=True)
    steam_id = db.Column(db.String(20), nullable=False, index=True)
    app_id = db.Column(db.Integer, nullable=False)
    game_name = db.Column(db.String(255))
    playtime_hours = db.Column(db.Float)
    genres = db.Column(db.Text) # Gunakan Text untuk menampung daftar genre yang panjang
    achievements = db.Column(db.Integer)

    def __init__(self, steam_id, app_id, game_name, playtime_hours, genres, achievements):
        self.steam_id = steam_id
        self.app_id = app_id
        self.game_name = game_name
        self.playtime_hours = playtime_hours
        self.genres = genres
        self.achievements = achievements