from src.controllers.TopicModelingController import AnalyzeApp
from src.controllers.AuthController import AuthApp

routes = [
  { "url": "/analyze", "name": AnalyzeApp },
  { "url": "/auth", "name": AuthApp},
]

