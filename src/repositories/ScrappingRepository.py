from src.models.Scrapping import Scrapping, db

class ScrappingRepository:
    def getAllScrapping(self):
        return Scrapping.query.all()

    def getScrappingBySteamId(self, steam_id):
        return Scrapping.query.filter_by(steam_id=steam_id).first()
    
    def createNewScrapping(self, data):
        newScrapping = Scrapping(
            Steam_id=data["Steam_ID"],
            Page=data["Page"],
            Game=data["Game"],
            Genre=data["Genre"],
            Rating=data["Rating"],
            Review=data["Review"],
            Playtime=data["Playtime"],
            PostedDate=data["PostedDate"]
        )
        db.session.add(newScrapping)
        db.session.commit()
        return newScrapping
