from src.models.SteamIDProses import SteamIDProses, db
from src.models.User import User
from sqlalchemy import func
from sqlalchemy.orm import joinedload

class SteamIDProsesRepository:
    def getAllSteamIDProses(self):
        return SteamIDProses.query.all()
    def getAllProsesByUserId(self, user_id):
        return SteamIDProses.query.filter_by(User_id=user_id).order_by(SteamIDProses.Proses_id.desc()).all()
    def getSteamIDProsesByProsesId(self, proses_id):
        return SteamIDProses.query.filter_by(Proses_id=proses_id).first()
    def getDetailProses(self, proses_id, user_id):
        return SteamIDProses.query.options(
                    joinedload(SteamIDProses.segmentasi_entries),
                    joinedload(SteamIDProses.topics)
                ).filter_by(Proses_id=proses_id, User_id=user_id).first()
    def createNewSteamIDProses(self,steam_ids, user_id):
        newSteamIDProses = SteamIDProses(
            Steam_ids=steam_ids,
            User_id=user_id
        )
        db.session.add(newSteamIDProses)
        return newSteamIDProses

    def getMaxProsesIdByUser(self, user_id):
        max_id = db.session.query(func.max(SteamIDProses.Proses_id))\
            .join(SteamIDProses.users)\
            .filter(User.user_id == user_id)\
            .scalar()
        return max_id