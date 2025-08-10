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
from sklearn.preprocessing import normalize
import numpy as np

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

    # Cukup satu fungsi load, yang lain bisa dihapus
    def _load_model(self):
    # Ganti path ke folder model yang baru
        MODEL_PATH = "public/models/player_topic_model2"
        print(f"🔍 Mencoba memuat model (safetensors) dari: {MODEL_PATH}")
        try:
            # Tidak perlu lagi argumen embedding_model saat load
            model = BERTopic.load(MODEL_PATH, embedding_model="all-MiniLM-L6-v2")
            print("✅ Model BERTopic (safetensors) berhasil dimuat dengan benar.")

            # (Opsional) Lakukan diagnosis cepat lagi untuk memastikan
            if hasattr(model, "__version__"):
                print(f"Versi BERTopic pada model: {model.__version__}")
            else:
                print("Atribut '__version__' tidak ditemukan.") # Seharusnya tidak terjadi lagi

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

    def _calculate_dominant_topic_per_game(self, reviews_df_with_results):
        """
        Menghitung topik dominan untuk setiap game berdasarkan kemiripan embedding.
        
        Args:
            reviews_df_with_results (pd.DataFrame): DataFrame yang berisi kolom 'Game', 'topics', dan 'embedding'.

        Returns:
            pd.DataFrame: DataFrame dengan 'Game', 'Dominant_Topic', dan 'Similarity_Score'.
        """
        print("🔍 Mengekstrak embedding dan centroid topik...")
        
        # 1. Dapatkan centroid topik dari model BERTopic yang sudah dilatih
        # Indeks array ini sesuai dengan ID topik
        topic_centroids = self.topic_model.topic_embeddings_
        
        # 2. Normalisasi centroid topik
        # Ini diperlukan untuk menghitung cosine similarity secara efisien menggunakan dot product
        normalized_centroids = normalize(topic_centroids)
        
        # 3. Ambil embedding dari ulasan yang baru diprediksi dan normalisasi
        review_embeddings = np.array(reviews_df_with_results['embedding'].tolist())
        normalized_review_embeddings = normalize(review_embeddings)

        # 4. Hitung cosine similarity antara setiap ulasan dan setiap centroid topik
        # Hasilnya adalah matriks di mana baris = ulasan, kolom = topik
        print("🧮 Menghitung matriks kemiripan...")
        similarity_matrix = np.dot(normalized_review_embeddings, normalized_centroids.T)
        
        # 5. Buat DataFrame dari hasil kemiripan
        topic_ids = self.topic_model.get_topic_info().Topic.tolist()
        sim_df = pd.DataFrame(similarity_matrix, columns=topic_ids)
        
        # Tambahkan nama game ke DataFrame kemiripan
        sim_df['Game'] = reviews_df_with_results['Game'].values
        
        # 6. Agregasi untuk mendapatkan rata-rata kemiripan per game-topik
        print("🔄 Mengagregasi kemiripan per game...")
        # Ubah format dari wide ke long
        melted_df = sim_df.melt(
            id_vars='Game', 
            var_name='topic', 
            value_name='similarity',
            # Abaikan topik outlier (-1) dari analisis dominan
            value_vars=[c for c in sim_df.columns if c != 'Game' and c != -1] 
        )
        
        # Hitung rata-rata kemiripan
        mean_sim_per_game = melted_df.groupby(['Game', 'topic'])['similarity'].mean().reset_index()
        
        # 7. (Opsional tapi direkomendasikan) Beri penalti pada topik yang terlalu umum
        topic_counts = mean_sim_per_game['topic'].value_counts(normalize=True)
        penalty_factor = 1 - topic_counts
        mean_sim_per_game['penalized_sim'] = mean_sim_per_game.apply(
            lambda row: row['similarity'] * penalty_factor.get(row['topic'], 1),
            axis=1
        )

        # 8. Temukan topik dengan skor tertinggi untuk setiap game
        print("🏆 Menentukan topik dominan...")
        # Temukan indeks baris dengan 'penalized_sim' maksimum untuk setiap game
        idx = mean_sim_per_game.groupby('Game')['penalized_sim'].idxmax()
        
        # Pilih baris tersebut untuk mendapatkan hasil akhir
        dominant_topics = mean_sim_per_game.loc[idx, ['Game', 'topic', 'similarity']].reset_index(drop=True)
        
        # Ubah nama kolom agar lebih jelas
        dominant_topics = dominant_topics.rename(columns={
            'topic': 'Dominant_Topic',
            'similarity': 'Similarity_Score'
        })
        
        return dominant_topics
        
    def createNewTopicModeling(self, steam_ids, userId,  steam_proses_obj):
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

            if steam_id_scrapping:
                print(f"🚀 Memulai scraping untuk Steam IDs: {steam_id_scrapping}")
                reviews = get_game_reviews(steam_id_scrapping)
                if reviews:
                    scrapingService.createNewScrapping(reviews)
                    for review in reviews:
                        results.append(review)
            print(f"Total reviews yang akan dianalisis: {len(results)}")
            reviews_df = pd.DataFrame(results)

            if reviews_df.empty:
                return self.failedOrSuccessRequest('success', 200, {'message': 'Tidak ada review yang ditemukan.'})

            reviews_df.dropna(subset=['Review'], inplace=True)
            reviews = reviews_df['Review'].tolist()
            cleaned_reviews = [preprocess_single_text(review) for review in reviews]

            # 1. Hasilkan embedding untuk ulasan yang sudah dibersihkan
            print(f"⚙️ Membuat embedding untuk {len(cleaned_reviews)} review baru...")
            new_embeddings = self.topic_model.embedding_model.embed(cleaned_reviews)

            # 2. Dapatkan prediksi topik menggunakan embedding yang sudah dibuat
            print(f"⚙️ Menganalisis topik dengan BERTopic...")
            topics, _ = self.topic_model.transform(cleaned_reviews, embeddings=new_embeddings)
            
            # 3. Tambahkan hasil ke DataFrame untuk analisis lebih lanjut
            reviews_df['topics'] = topics
            reviews_df['embedding'] = list(new_embeddings) # Simpan embedding sebagai list dalam kolom

            # 4. Panggil metode baru untuk menghitung dan menyimpan topik dominan
            print("\n📊 Memulai proses perhitungan topik dominan per game...")
            dominant_topic_df = self._calculate_dominant_topic_per_game(reviews_df)
            
            # Simpan hasilnya ke file CSV
            output_path = "dominant_topic_per_game.csv"
            dominant_topic_df.to_csv(output_path, index=False)
            print(f"✅ Topik dominan per game berhasil disimpan di: {output_path}\n")

            steam_proses = steam_proses_obj
            
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
