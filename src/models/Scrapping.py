from src.server.main import db,main_app
from src.config.database import generateDatabase

class Scrapping(db.Model):
    __tablename__ = 'scrapping'
    Scrapping_id = db.Column(db.Integer, primary_key=True)
    Steam_id = db.Column(db.String(255))
    Page = db.Column(db.Integer)    
    Game = db.Column(db.String(255))
    Genre = db.Column(db.String(255))
    Rating =  db.Column(db.String(255))
    Review = db.Column(db.Text)
    Playtime = db.Column(db.String(255))
    PostedDate = db.Column(db.String(255))

    def __init__(self, Steam_id, Page, Game, Genre, Rating, Review, Playtime, PostedDate):
        self.Steam_id = Steam_id
        self.Page = Page
        self.Game = Game
        self.Genre = Genre
        self.Rating = Rating
        self.Review = Review
        self.Playtime = Playtime
        self.PostedDate = PostedDate
