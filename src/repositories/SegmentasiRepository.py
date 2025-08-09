from src.models.Segmentasi import Segmentasi, db

class SegmentasiRepository:
    def getAllSegmentasi(self):
        return Segmentasi.query.all()

    def getSegmentasiByProsesId(self, proses_id):
        return Segmentasi.query.filter_by(Proses_id=proses_id).first()

    def createNewSegmentasi(self, data):
        newSegmentasi = Segmentasi(
            Proses_id=data["Proses_id"],
            Arketipe=data["Arketipe"]
        )
        db.session.add(newSegmentasi)
        db.session.commit()
        return newSegmentasi
