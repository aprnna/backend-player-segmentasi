from src.models.TopicModeling import TopicModeling, db
from src.models.SteamIDProses import SteamIDProses 
class TopicModelingRepository:
    def getAllTopicModeling(self):
        return TopicModeling.query.all()
    def getAllTopicModelingByProsesId(self, proses_id):
        return TopicModeling.query.filter_by(Proses_id=proses_id).all()
    def getAllTopicModelingByProsesId(self, proses_id):
        proses = SteamIDProses.query.get(proses_id)
        
        if proses:
            return proses.topics  # SQLAlchemy secara ajaib akan mengambil semua topik terkait
        else:
            return []
    def getTopicByClusterId(self, cluster_id):
        return TopicModeling.query.filter_by(Cluster=cluster_id).all()
    def createNewTopicModeling(self,keyword, cluster):
        newTopicModeling = TopicModeling(
            Keyword=keyword,
            Cluster=cluster
        )
        db.session.add(newTopicModeling)
        return newTopicModeling

