from abc import ABC, abstractmethod
class Service(ABC):
  @staticmethod
  @abstractmethod
  def failedOrSuccessRequest(self, status, code, data):
    pass