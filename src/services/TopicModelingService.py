from  src.services.Service import Service
from src.config.database import db
from src.repositories.TopicModelingRepository import TopicModelingRepository
from src.repositories.UserRepository import UserRepository
from src.repositories.SteamIDProsesRepository import SteamIDProsesRepository
from src.services.ScrappingService import ScrappingService
from src.repositories.ScrappingRepository import ScrappingRepository
from src.utils.convert import queryResultToDict
from src.utils.errorHandler import errorHandler
from src.utils.preprocessing.text_processor import preprocess_single_text
from src.utils.scraping.steam_review import get_game_reviews
from bertopic import BERTopic
import pandas as pd

userRepository = UserRepository()
steamIDProsesRepository = SteamIDProsesRepository()
topic_modeling_repository = TopicModelingRepository()
scrapingService = ScrappingService()
scrapingRepository = ScrappingRepository()

class TopicModelingService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }

    def __init__(self):
        self.topic_model = self._load_model()

    def _load_model(self):
        MODEL_PATH = 'public/models/player_topic_model'
        print(f"🔍 Mencoba memuat model dari: {MODEL_PATH}")
        try:
            model = BERTopic.load(MODEL_PATH, embedding_model="all-MiniLM-L6-v2")
            print("✅ Model BERTopic berhasil dimuat.")
            return model
        except Exception as e:
            print(f"❌ Gagal memuat model BERTopic: {e}")
            return None
    
    def getTopicModelling(self):
        try:
            topic_modeling = topic_modeling_repository.getAllTopicModeling()
            if not topic_modeling:
                return self.failedOrSuccessRequest('failed', 404, 'No topic modeling data found')
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(topic_modeling))
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))

    def getTopicModelingByProsesId(self, topic_modeling_id):
        try:
            topic_modeling = topic_modeling_repository.getTopicModelingByProsesId(topic_modeling_id)
            if not topic_modeling:
                return self.failedOrSuccessRequest('failed', 404, 'Topic modeling data not found')
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(topic_modeling))
        except ValueError as e:
            return self.failedOrSuccessRequest('failed', 500, errorHandler(e.errors()))
        
    def createNewTopicModeling(self, steam_ids, userId):
        if not self.topic_model:
            return self.failedOrSuccessRequest('failed', 503, {'message': 'Model analisis tidak tersedia.'})
        try:
            # 2. Validasi Pengguna
            user = userRepository.getUserById(userId)
            if not user:
                return self.failedOrSuccessRequest('failed', 404, {'message': f'User dengan id {userId} tidak ditemukan.'})
            results = []
            steam_id_scrapping = []
            for steam_id in steam_ids:
                getReviewsScrapping = scrapingService.getAllScrappingBySteamId(steam_id)
                if getReviewsScrapping.get('status') == 'failed':
                    print(f"Tidak ada data review di database: {steam_id}")
                    steam_id_scrapping.append(steam_id)
                else:
                    reviews = getReviewsScrapping.get('data', [])
                    for review in reviews:
                        results.append(review)
            print(results)

            if steam_id_scrapping:
                print(f"🚀 Memulai scraping untuk Steam IDs: {steam_id_scrapping}")
                reviews = get_game_reviews(steam_id_scrapping)
                if reviews:
                    scrapingService.createNewScrapping(reviews)
                    for review in reviews:
                        results.append(review)
            print(f"Total reviews yang akan dianalisis: {len(results)}")
            reviews_df = pd.DataFrame(results)
            # getReviewsScrapping = scrapingService.getAllScrappingBySteamId(steam_ids)
            # if getReviewsScrapping.get('status') == 'failed':
            #     print(f"🚀 Memulai scraping untuk Steam IDs: {steam_ids}")
            #     reviews = get_game_reviews(steam_ids)
            #     reviews_df = pd.DataFrame(reviews)
            #     list_of_reviews = reviews_df.to_dict('records')
            #     scrapingService.createNewScrapping(list_of_reviews)
            # else:
            #     reviews = getReviewsScrapping.get('data', [])
            # reviews_df = pd.DataFrame(reviews)

            if reviews_df.empty:
                return self.failedOrSuccessRequest('success', 200, {'message': 'Tidak ada review yang ditemukan.'})

            reviews_df.dropna(subset=['Review'], inplace=True)
            reviews = reviews_df['Review'].tolist()
            cleaned_reviews = [preprocess_single_text(review) for review in reviews]
            steamId_as_string = ", ".join(steam_ids)
            steam_proses = steamIDProsesRepository.createNewSteamIDProses(steam_ids=steamId_as_string, user_id=userId)
            
            print(f"⚙️ Menganalisis {len(cleaned_reviews)} review dengan BERTopic...")
            # Dapatkan prediksi topik untuk setiap review
            topics, _ = self.topic_model.transform(cleaned_reviews)
            reviews_df['topics'] = topics
            reviews_df.to_csv('predicted_reviews.csv', index=False)  # Simpan hasil prediksi
            # 4. Proses Penyimpanan ke Database dengan Relasi
            unique_topics = sorted(list(set(topics)))
            results = []
            for topic_id in unique_topics:
                # Abaikan outlier
                if topic_id == -1:
                    continue
                # Dapatkan keywords untuk topic_id yang diprediksi
                topic_info = self.topic_model.get_topic(topic_id)
                if not topic_info: continue
                # 1. Kumpulkan semua keyword untuk satu topik ke dalam list
                all_keywords = [word for word, score in topic_info]
                results.append({
                    'topic_id': int(topic_id),
                    'keywords': all_keywords,
                    'number_of_reviews': topics.tolist().count(topic_id),
                })
                # 2. Gabungkan menjadi satu string (misalnya, dipisahkan koma)
                #    Pastikan kolom 'Keyword' di model TopicModeling cukup besar (misal: db.Text)
                keywords_as_string = ", ".join(all_keywords)
                topic_entry = topic_modeling_repository.createNewTopicModeling(keyword=keywords_as_string, cluster=topic_id)
                # 4. Hubungkan satu entri topik ini ke steam_proses
                if topic_entry not in steam_proses.topics:
                    steam_proses.topics.append(topic_entry)
                db.session.add(steam_proses)
            db.session.commit()
            return self.failedOrSuccessRequest('success', 201, results)

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error dalam analyze_and_save_steam_reviews: {e}")
            return self.failedOrSuccessRequest('failed', 500, {'message': 'Terjadi kesalahan internal pada server.'})
