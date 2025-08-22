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
from src.utils.scraping.steam_review import get_game_reviews_parallel
from bertopic import BERTopic
import pandas as pd
from sklearn.preprocessing import normalize
import numpy as np
from src.utils.job_manager import job_status

import threading
import os
import gc
import psutil
import threading
import time
from contextlib import contextmanager
import pickle
import hashlib
from pathlib import Path
import tempfile
import logging

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

    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.MODEL_PATH = "public/models/player_topic_model2"
            self.CACHE_DIR = "/tmp/topic_cache"
            self.MAX_MEMORY_PERCENT = 75  # Gunakan maksimal 75% memory
            self.BATCH_SIZE = 20  # Sangat kecil untuk VPS 1GB
            
            # Buat cache directory
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            
            # Setup logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024)
        }
    
    def _emergency_cleanup(self):
        """Emergency cleanup untuk free memory"""
        self.logger.warning("🚨 Emergency memory cleanup...")
        
        # Clear Python caches
        gc.collect()
        
        # Clear temporary files
        try:
            os.system(f"rm -rf {self.CACHE_DIR}/*")
        except:
            pass
            
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
    
    @contextmanager
    def memory_guard(self, operation_name="operation"):
        """Context manager untuk monitor memory usage"""
        memory_before = self._get_memory_usage()
        self.logger.info(f"🧠 Starting {operation_name} - Memory: {memory_before['percent']:.1f}%")
        
        try:
            if memory_before['percent'] > self.MAX_MEMORY_PERCENT:
                self._emergency_cleanup()
                
            yield
            
        except MemoryError as e:
            self.logger.error(f"❌ Memory error during {operation_name}: {e}")
            self._emergency_cleanup()
            raise
        finally:
            memory_after = self._get_memory_usage()
            self.logger.info(f"🧠 Finished {operation_name} - Memory: {memory_after['percent']:.1f}%")
    
    def _load_model_on_demand(self):
        """Load model hanya saat benar-benar dibutuhkan"""
        with self.memory_guard("model_loading"):
            try:
                # Cek apakah model file ada
                if not os.path.exists(self.MODEL_PATH):
                    raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")
                
                # Import hanya saat dibutuhkan untuk save memory
                from bertopic import BERTopic
                
                # Load dengan embedding model paling ringan
                self.logger.info("📥 Loading ultra-light model...")
                model = BERTopic.load(
                    self.MODEL_PATH,
                    embedding_model="paraphrase-MiniLM-L3-v2"  # Hanya 17MB
                )
                
                return model
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load model: {e}")
                raise

    def _process_reviews_in_batches(self, reviews, batch_size=None):
        """Process reviews dalam batch sangat kecil"""
        if batch_size is None:
            batch_size = self.BATCH_SIZE
            
        results = []
        total_batches = (len(reviews) + batch_size - 1) // batch_size
        
        self.logger.info(f"📦 Processing {len(reviews)} reviews in {total_batches} batches")
        
        for i in range(0, len(reviews), batch_size):
            batch_num = (i // batch_size) + 1
            batch = reviews[i:i + batch_size]
            
            # with self.memory_guard(f"batch_{batch_num}"):
            # Load model untuk batch ini
            model = self._load_model_on_demand()
            
            try:
                # Process batch
                self.logger.info(f"⚙️ Processing batch {batch_num}/{total_batches}")
                
                # Generate embeddings
                embeddings = model.embedding_model.embed(batch)
                
                # Get topics
                topics, _ = model.transform(batch, embeddings=embeddings)
                
                # Store results
                batch_results = {
                    'topics': topics.tolist(),
                    'reviews': batch,
                    'batch_num': batch_num
                }
                results.append(batch_results)
                
                # Cleanup immediately after each batch
                del model, embeddings, topics
                # gc.collect()
                
                self.logger.info(f"✅ Batch {batch_num} completed")
                
            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_num}: {e}")
                # Cleanup pada error
                try:
                    del model
                except:
                    pass
                # gc.collect()
                raise
        
        return results
    def _cache_results(self, results, cache_key):
        """Cache results untuk menghindari reprocessing"""
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"💾 Results cached to {cache_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cache results: {e}")
    
    def _load_cached_results(self, cache_key):
        """Load cached results jika ada"""
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                self.logger.info(f"💾 Loaded cached results from {cache_file}")
                return results
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load cache: {e}")
        return None
    def getAllTopicsByProsesId(self, proses_id):
        try:
            # Panggil metode repository yang sudah diperbaiki
            list_of_topics = topic_modeling_repository.getAllTopicModelingByProsesId(proses_id)
            if not list_of_topics:
                return self.failedOrSuccessRequest(
                    'failed', 
                    404, 
                    {'message': f'Tidak ada topik yang ditemukan untuk Proses ID: {proses_id}'}
                )
            return self.failedOrSuccessRequest('success', 200, queryResultToDict(list_of_topics))

        except Exception as e:
            print(f"❌ Error saat mengambil topik berdasarkan Proses ID: {e}")
            return self.failedOrSuccessRequest('failed', 500, {'message': 'Terjadi kesalahan internal.'})

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

    # def _calculate_dominant_topic_per_game(self, reviews_df_with_results):
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
        
    # def createNewTopicModeling(self, steam_ids, userId,  steam_proses_obj, unique_process_dir):
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
            dominant_output_path = os.path.join(unique_process_dir, "dominant_topic_per_game.csv")
            dominant_topic_df.to_csv(dominant_output_path, index=False)
            print(f"✅ Topik dominan per game berhasil disimpan di: {dominant_output_path}\n")

            steam_proses = steam_proses_obj
            review_output_path = os.path.join(unique_process_dir, "predicted_reviews.csv")
            reviews_df.to_csv(review_output_path, index=False)  # Simpan hasil prediksi
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
    
    def _calculate_dominant_topic_per_game(self, reviews_df_with_results):
        """
        Menghitung topik dominan untuk setiap game berdasarkan kemiripan embedding.
        Optimized untuk VPS 1GB RAM.
        """
        print("🔍 Mengekstrak embedding dan centroid topik...")
        
        with self.memory_guard("dominant_topic_calculation"):
            try:
                import numpy as np
                from sklearn.preprocessing import normalize
                
                # Load model untuk dapatkan centroid
                model = self._load_model_on_demand()
                
                # 1. Dapatkan centroid topik dari model BERTopic
                topic_centroids = model.topic_embeddings_
                normalized_centroids = normalize(topic_centroids)
                
                # 2. Ambil embedding dari ulasan (dalam batch untuk save memory)
                embeddings_list = reviews_df_with_results['embedding'].tolist()
                review_embeddings = np.array(embeddings_list)
                normalized_review_embeddings = normalize(review_embeddings)
                
                # 3. Hitung cosine similarity dalam batch untuk avoid memory spike
                print("🧮 Menghitung matriks kemiripan...")
                similarity_matrix = np.dot(normalized_review_embeddings, normalized_centroids.T)
                
                # 4. Buat DataFrame kemiripan
                topic_ids = model.get_topic_info().Topic.tolist()
                
                import pandas as pd
                sim_df = pd.DataFrame(similarity_matrix, columns=topic_ids)
                sim_df['Game'] = reviews_df_with_results['Game'].values
                
                # Cleanup arrays untuk save memory
                del review_embeddings, normalized_review_embeddings, similarity_matrix
                gc.collect()
                
                # 5. Agregasi untuk mendapatkan rata-rata kemiripan per game-topik
                print("🔄 Mengagregasi kemiripan per game...")
                melted_df = sim_df.melt(
                    id_vars='Game', 
                    var_name='topic', 
                    value_name='similarity',
                    value_vars=[c for c in sim_df.columns if c != 'Game' and c != -1] 
                )
                
                # Hitung rata-rata kemiripan
                mean_sim_per_game = melted_df.groupby(['Game', 'topic'])['similarity'].mean().reset_index()
                
                # 6. Penalty untuk topik umum (optional, dimatikan untuk save memory di VPS 1GB)
                # topic_counts = mean_sim_per_game['topic'].value_counts(normalize=True)
                # penalty_factor = 1 - topic_counts
                # mean_sim_per_game['penalized_sim'] = mean_sim_per_game.apply(
                #     lambda row: row['similarity'] * penalty_factor.get(row['topic'], 1),
                #     axis=1
                # )
                
                # 7. Temukan topik dengan skor tertinggi untuk setiap game
                print("🏆 Menentukan topik dominan...")
                idx = mean_sim_per_game.groupby('Game')['similarity'].idxmax()  # Use similarity directly
                
                dominant_topics = mean_sim_per_game.loc[idx, ['Game', 'topic', 'similarity']].reset_index(drop=True)
                dominant_topics = dominant_topics.rename(columns={
                    'topic': 'Dominant_Topic',
                    'similarity': 'Similarity_Score'
                })
                
                # Cleanup
                del model, sim_df, melted_df, mean_sim_per_game
                gc.collect()
                
                return dominant_topics
                
            except Exception as e:
                self.logger.error(f"❌ Error calculating dominant topics: {e}")
                # Emergency cleanup
                try:
                    del model
                except:
                    pass
                gc.collect()
                raise e

    def createNewTopicModeling(self, steam_ids, userId, steam_proses_obj, unique_process_dir, job_id):
        """Main method dengan optimasi extreme untuk VPS 1GB - Complete Implementation"""
        
        # Pre-flight check
        # self._emergency_cleanup()
        # memory_info = self._get_memory_usage()
        # if memory_info['available_mb'] < 5000:  # Kurang dari 5000MB
        #     return self.failedOrSuccessRequest('failed', 503, 
        #         {'message': f'Insufficient memory. Available: {memory_info["available_mb"]:.0f}MB, Required: >5000MB'})
        
        try:
            # Import dependencies
            from src.repositories.UserRepository import UserRepository
            from src.services.ScrappingService import ScrappingService
            from src.repositories.ScrappingRepository import ScrappingRepository
            from src.repositories.TopicModelingRepository import TopicModelingRepository
            from src.utils.preprocessing.text_processor import preprocess_single_text
            from src.utils.scraping.steam_review import get_game_reviews_parallel
            from src.config.database import db
            import pandas as pd
            import os
            
            userRepository = UserRepository()
            scrapingService = ScrappingService()
            topic_modeling_repository = TopicModelingRepository()
            
            # Generate cache key berdasarkan steam_ids
            # cache_key = hashlib.md5(','.join(sorted(steam_ids)).encode()).hexdigest()           
            
            # Cek cache terlebih dahulu
            # cached_results = self._load_cached_results(cache_key)
            # if cached_results:
            #     self.logger.info("🎯 Using cached results")
            #     return self.failedOrSuccessRequest('success', 200, cached_results)
            
            # 2. Validasi Pengguna
            user = userRepository.getUserById(userId)
            if not user:
                return self.failedOrSuccessRequest('failed', 404, 
                    {'message': f'User dengan id {userId} tidak ditemukan.'})
            
            # 3. Kumpulkan reviews dengan batching
            self.logger.info("🔍 Starting review collection...")
            results = []
            steam_id_scrapping = []
            job_status[job_id].update({
                'status': 'processing',
                'progress': 30,
                'message': "Collecting reviews..."
            })
            # Process steam_ids in smaller groups untuk avoid memory spike
            steam_id_batches = [steam_ids[i:i+5] for i in range(0, len(steam_ids), 5)]

            for batch_steam_ids in steam_id_batches:
                # with self.memory_guard("scraping_batch"):
                for steam_id in batch_steam_ids:
                    getReviewsScrapping = scrapingService.getAllScrappingBySteamId(steam_id)
                    if getReviewsScrapping.get('status') == 'failed':
                        print(f"Tidak ada data review di database: {steam_id}")
                        steam_id_scrapping.append(steam_id)
                    else:
                        reviews = getReviewsScrapping.get('data', [])
                        for review in reviews:  # Max 30 reviews per game
                            results.append(review)
                    # gc.collect()
            
            # Scraping jika diperlukan
            if steam_id_scrapping:
                print(f"🚀 Memulai scraping untuk Steam IDs: {steam_id_scrapping}")
                # with self.memory_guard("scraping_new"):
                reviews = get_game_reviews_parallel(steam_id_scrapping)
                if reviews:
                    scrapingService.createNewScrapping(reviews)
                    for review in reviews:  # Limit scraping results
                        results.append(review)
            
            print(f"Total reviews yang akan dianalisis: {len(results)}")
            if len(results) == 0:
                self.logger.warning("⚠️ Tidak ada review yang ditemukan untuk dianalisis")
            # Limit total reviews untuk VPS 1GB
            # MAX_REVIEWS = 150  # Reduced limit untuk VPS 1GB
            # if len(results) > MAX_REVIEWS:
            #     results = results[:MAX_REVIEWS]
            #     self.logger.warning(f"⚠️ Limited to {MAX_REVIEWS} reviews due to memory constraints")
            
            if not results:
                return self.failedOrSuccessRequest('success', 200, 
                    {'message': 'Tidak ada review yang ditemukan.'})
            
            job_status[job_id].update({
                'status': 'processing',
                'progress': 40,
                'message': "Preprocessing reviews..."
            })
            # 4. Preprocessing dengan memory management
            # with self.memory_guard("preprocessing"):
            reviews_df = pd.DataFrame(results)
            reviews_df.dropna(subset=['Review'], inplace=True)
            reviews = reviews_df['Review'].tolist()
            
            # Preprocess dalam batch kecil
            cleaned_reviews = []
            for i in range(0, len(reviews), 10):
                batch = reviews[i:i+10]
                batch_cleaned = [preprocess_single_text(review) for review in batch]
                cleaned_reviews.extend(batch_cleaned)
                    
                    # if i % 30 == 0:  # Cleanup setiap 30 reviews
                    #     gc.collect()
            
            # 5. Topic modeling dengan ultra-batch processing
            print(f"⚙️ Membuat embedding untuk {len(cleaned_reviews)} review baru...")
            job_status[job_id].update({
                'status': 'processing',
                'progress': 45,
                'message': "Creating embeddings..."
            })  

            model = self._load_model_on_demand()
            all_topics = []
            all_embeddings = []
            
            try:
                BATCH_SIZE = 50  
                for i in range(0, len(cleaned_reviews), BATCH_SIZE):
                    batch = cleaned_reviews[i:i+BATCH_SIZE]
                    
                    with self.memory_guard(f"topic_batch_{i//BATCH_SIZE + 1}"):
                        print(f"⚙️ Processing batch {i//BATCH_SIZE + 1}/{(len(cleaned_reviews)-1)//BATCH_SIZE + 1}")
                        
                        # Generate embeddings
                        new_embeddings = model.embedding_model.embed(batch)
                        
                        # Get topics
                        topics, _ = model.transform(batch, embeddings=new_embeddings)
                        
                        # Store results
                        all_topics.extend(topics.tolist())
                        all_embeddings.extend(list(new_embeddings))
                        
                        # Cleanup immediately
                        del new_embeddings, topics
                        gc.collect()
                
                # 6. Tambahkan hasil ke DataFrame
                reviews_df = reviews_df.iloc[:len(all_topics)].copy()  # Ensure same length
                reviews_df['topics'] = all_topics
                reviews_df['embedding'] = all_embeddings
                
                job_status[job_id].update({
                    'status': 'processing',
                    'progress': 55,
                    'message': "Calculating dominant topics..."
                })
                # 7. Hitung topik dominan per game
                print("\n📊 Memulai proses perhitungan topik dominan per game...")
                dominant_topic_df = self._calculate_dominant_topic_per_game(reviews_df)
                
                # Simpan hasil ke file
                dominant_output_path = os.path.join(unique_process_dir, "dominant_topic_per_game.csv")
                dominant_topic_df.to_csv(dominant_output_path, index=False)
                print(f"✅ Topik dominan per game berhasil disimpan di: {dominant_output_path}")
                
                review_output_path = os.path.join(unique_process_dir, "predicted_reviews.csv")
                reviews_df.to_csv(review_output_path, index=False)
                
                # 8. Simpan ke database
                steam_proses = steam_proses_obj
                unique_topics = sorted(list(set(all_topics)))
                results = []
                
                for topic_id in unique_topics:
                    # Abaikan outlier
                    if topic_id == -1:
                        continue
                    
                    # Dapatkan keywords
                    topic_info = model.get_topic(topic_id)
                    if not topic_info: 
                        continue
                    
                    # Limit keywords untuk save memory
                    all_keywords = [word for word, score in topic_info]
                    results.append({
                        'topic_id': int(topic_id),
                        'keywords': all_keywords,
                        'number_of_reviews': all_topics.count(topic_id),
                    })
                    
                    # Save to database
                    keywords_as_string = ", ".join(all_keywords)
                    topic_entry = topic_modeling_repository.createNewTopicModeling(
                        keyword=keywords_as_string, 
                        cluster=topic_id
                    )
                    
                    if topic_entry not in steam_proses.topics:
                        steam_proses.topics.append(topic_entry)
                    db.session.add(steam_proses)
                
                db.session.commit()
                
            finally:
                # Cleanup model dan variables
                try:
                    del model
                except:
                    pass
                gc.collect()
            
            # Cache results
            # self._cache_results(results, cache_key)
            job_status[job_id].update({
                'status': 'processing',
                'progress': 65,
                'message': "Topic modeling completed successfully."
            })

            self.logger.info("✅ Topic modeling completed successfully")
            return self.failedOrSuccessRequest('success', 201, results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                db.session.rollback()
            except:
                pass
            self.logger.error(f"❌ Error in topic modeling: {e}")
            self._emergency_cleanup()
            return self.failedOrSuccessRequest('failed', 500, 
                {'message': f'Terjadi kesalahan internal pada server: {str(e)}'})
    