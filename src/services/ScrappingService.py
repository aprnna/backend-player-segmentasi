from  src.services.Service import Service
from src.repositories.ScrappingRepository import ScrappingRepository
from src.utils.convert import queryResultToDict
from src.utils.errorHandler import errorHandler

scrapping_repository = ScrappingRepository()

class ScrappingService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }
    
    def getScrapping(self):
        try:
            scrappings = scrapping_repository.getAllScrapping()
            if not scrappings:
                return self.failedOrSuccessRequest('failed', 404, 'No scrapping data found')
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(scrappings))
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))
        
    def getScrappingBySteamId(self, scrapping_id):
        try:
            scrapping = scrapping_repository.getScrappingBySteamId(scrapping_id)
            if not scrapping:
                return self.failedOrSuccessRequest('failed', 404, 'Scrapping data not found')
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(scrapping))
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))
        
    def getAllScrappingBySteamId(self, steam_id):
        try:
            scrappings = scrapping_repository.getAllScrappingBySteamId(steam_id)
            if not scrappings:
                return self.failedOrSuccessRequest('failed', 404, 'No scrapping data found for this Steam ID')
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(scrappings))
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))
        
    def createNewScrapping(self, datas):
        try:
            saved_records = []
            for data in datas:
                # 'data' sudah merupakan dictionary
                # pastikan semua kolom ada
                required_columns = ["Steam_ID", "Page", "Game", "Genre", "Rating", "Review", "Playtime", "PostedDate"]
                if not all(col in data for col in required_columns):
                    continue
                new_scrapping = scrapping_repository.createNewScrapping(data)
                saved_records.append(new_scrapping)
            
            if not saved_records:
                return self.failedOrSuccessRequest('success', 200, {'message': 'No new reviews to save.'})

            return self.failedOrSuccessRequest(
                'success', 
                201, 
                {'message': f'{len(saved_records)} records created successfully.'}
            )
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))
        