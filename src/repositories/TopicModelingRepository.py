from src.models.TopicModeling import TopicModeling, db

class TopicModelingRepository:
    def getAllTopicModeling(self):
        return TopicModeling.query.all()

    def getTopicModelingByProsesId(self, proses_id):
        return TopicModeling.query.filter_by(Proses_id=proses_id).first()

    def createNewTopicModeling(self,keyword, cluster):
        newTopicModeling = TopicModeling(
            Keyword=keyword,
            Cluster=cluster
        )
        db.session.add(newTopicModeling)
        return newTopicModeling

