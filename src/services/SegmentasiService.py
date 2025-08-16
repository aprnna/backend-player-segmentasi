from  src.services.Service import Service
from src.utils.convert import queryResultToDict
from src.utils.errorHandler import errorHandler
from src.repositories.SegmentasiRepository import SegmentasiRepository
from src.repositories.TopicModelingRepository import TopicModelingRepository
from src.repositories.PlayerGameDataRepository import PlayerGameDataRepository
from src.utils.scraping.steam_data import get_steam_id_data
import pandas as pd
import numpy as np
import glob
import os
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import nnls
from kneed import KneeLocator
import matplotlib.pyplot as plt
import umap
import warnings
import json
from difflib import SequenceMatcher
from collections import Counter
import requests
import ast
from src.config.config import GROQ_API_KEY

segmentasiRepository = SegmentasiRepository()
topic_modeling_repository = TopicModelingRepository()
player_game_data_repository = PlayerGameDataRepository()

class SegmentationService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }
    def get_summary_by_proses_id(self, proses_id):
        data = segmentasiRepository.get_summary_by_proses_id(proses_id)
        if not data:
            return self.failedOrSuccessRequest('error', 404, {"message": "Ringkasan untuk Proses ID tersebut tidak ditemukan."})
        return self.failedOrSuccessRequest('success', 200, queryResultToDict(data))
    def _generate_archetype_interpretations(self, characteristics_data, proses_id):
        """
        Menggunakan API untuk menghasilkan interpretasi kualitatif dari data 
        karakteristik kuantitatif arketipe.
        """
        print("🤖 Menghubungi API untuk menghasilkan interpretasi arketipe...")
        
        try:
            # 1. Buat topic mapping dari database
            list_of_topics = topic_modeling_repository.getAllTopicModelingByProsesId(proses_id)
            if not list_of_topics:
                print("⚠️ Peringatan: Tidak ada topik yang ditemukan untuk Proses ID ini. Melewatkan interpretasi LLM.")
                return []
            
            topic_mapping = {
                str(t.Cluster): [kw.strip() for kw in t.Keyword.split(',')][:3] for t in list_of_topics
            }
            
            # 2. Siapkan payload untuk API
            api_payload = {
                "characteristics_data": characteristics_data,
                "topic_mapping": topic_mapping
            }
            
            # 3. Tentukan URL API - sesuaikan dengan konfigurasi Anda
            # Bisa dari environment variable atau hardcode untuk testing
            API_BASE_URL = os.getenv('ARCHETYPE_API_URL', 'http://localhost:5000')
            api_endpoint = f"{API_BASE_URL}/api/archetype/interpret"
            
            print(f"🌐 Mengirim request ke: {api_endpoint}")
            
            # 4. Kirim request ke API
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                api_endpoint, 
                headers=headers, 
                json=api_payload, 
                timeout=60
            )
            
            # 5. Handle response
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                print("✅ Interpretasi berhasil dibuat melalui API.")
                return result.get("data", [])
            else:
                print(f"⚠️ API mengembalikan error: {result.get('error', 'Unknown error')}")
                # Fallback ke data kosong jika API gagal
                return result.get("data", [])
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Gagal terhubung ke API interpretasi arketipe: {e}")
            print("💡 Pastikan API server berjalan di URL yang benar")
            return []
        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout saat menghubungi API interpretasi arketipe: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ Error saat menghubungi API interpretasi arketipe: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error saat menghubungi API: {e}")
            return []
    def _generate_archetype_interpretations_with_llm(self, characteristics_data, proses_id):
        """
        Menggunakan LLM untuk menghasilkan interpretasi kualitatif dari data 
        karakteristik kuantitatif arketipe.
        """
        print("🤖 Menghubungi LLM untuk menghasilkan interpretasi arketipe...")
        
        # 1. Ambil API Key dari environment variable
        API_KEY =  GROQ_API_KEY
        if not API_KEY:
            print("⚠️ Peringatan: GROQ_API_KEY tidak ditemukan. Melewatkan interpretasi LLM.")
            return [] # Kembalikan list kosong jika tidak ada API Key

        ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

        # 2. Buat topic mapping dari database, bukan CSV
        list_of_topics = topic_modeling_repository.getAllTopicModelingByProsesId(proses_id)
        if not list_of_topics:
            print("⚠️ Peringatan: Tidak ada topik yang ditemukan untuk Proses ID ini. Melewatkan interpretasi LLM.")
            return []
        topic_mapping = {
            t.Cluster: [kw.strip() for kw in t.Keyword.split(',')][:3] for t in list_of_topics
        }

        # 3. Bangun prompt (sama seperti skrip Anda)
        system_msg = {
            "role": "system",
            "content": (
                "Anda adalah seorang Analis Data berpengalaman dan pakar perilaku pemain game. "
                "Tugas Anda adalah menafsirkan data JSON untuk setiap arketipe pemain berdasarkan fitur kuantitatif "
                "dan topik dominan yang dihasilkan dari analisis review game.\n\n"
                "FORMAT OUTPUT:\n"
                "Jawaban harus berupa array JSON tanpa teks tambahan. "
                "Setiap elemen berisi:\n"
                "  - \"arketipe\": nama key arketipe (contoh: 'Arketipe 1').\n"
                "  - \"fitur_kuantitatif\": ringkasan angka aktual.\n"
                "  - \"topik_dominan\": interpretasi makna topik dalam 1 frasa singkat, bukan sekadar kata kunci.\n"
                "  - \"interpretasi\": 1–2 kalimat berbahasa Indonesia yang mencakup angka aktual, makna topik, dan opini karakter pemain.\n\n"
                "ATURAN PENTING:\n"
                "- Setiap 'interpretasi' harus unik, tidak boleh terlalu mirip dengan arketipe lain.\n"
                "- Variasikan fokus insight (misalnya: kompetitif, sosial, eksploratif, artistik, dll).\n\n"
                "MAPPING TOPIK:\n"
                f"{json.dumps(topic_mapping, ensure_ascii=False)}\n\n"
                "RESPOND ONLY WITH A JSON ARRAY, tanpa teks tambahan."
            )
        }
        
        user_msg = {"role": "user", "content": json.dumps(characteristics_data, ensure_ascii=False)}

        # 4. Kirim request ke API
        try:
            payload = {
                "model": "llama3-8b-8192",
                "messages": [system_msg, user_msg],
                "temperature": 0.7, "max_tokens": 2048
            }
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            print(f"Raw response from LLM: {raw}")
            parsed = None
            try:
                # Coba cara paling sederhana dulu
                start, end = raw.find('['), raw.rfind(']')
                if start != -1 and end != -1:
                    json_text = raw[start:end+1]
                    parsed = json.loads(json_text)
                else: # Jika tidak ditemukan kurung siku, coba parse seluruh teks
                    parsed = json.loads(raw)

            except json.JSONDecodeError as e:
                print(f"⚠️ Gagal parse JSON standar. Mencoba perbaikan... Error: {e}")
                # Strategi perbaikan: hapus koma di akhir (trailing comma)
                # dan coba lagi. Ini adalah kesalahan umum dari LLM.
                cleaned_text = json_text.strip()
                # Hapus ```json, ```, dan newline di awal/akhir
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                
                # Coba parse lagi setelah dibersihkan
                try:
                    parsed = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # Jika masih gagal, lemparkan error agar bisa ditangkap di blok utama
                    print(f"❌ Gagal mem-parse respons dari LLM bahkan setelah dibersihkan.")
                    print("--- RAW Text yang Gagal di-Parse ---")
                    print(raw)
                    print("------------------------------------")
                    raise # Lemparkan error asli agar pipeline berhenti

            if not parsed:
                raise ValueError("Hasil parsing JSON kosong.")


            # 6. Post-processing (sama seperti skrip Anda)
            def is_similar(a, b, threshold=0.85):
                return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
            
            unique_interpretations = []
            for item in parsed:
                interp = item.get("interpretasi", "")
                if any(is_similar(interp, seen) for seen in unique_interpretations):
                    item["interpretasi"] += " Karakter pemain ini juga memiliki preferensi yang khas."
                unique_interpretations.append(interp)
            
            print("✅ Interpretasi berhasil dibuat oleh LLM.")
            return parsed

        except requests.exceptions.RequestException as e:
            print(f"❌ Gagal menghubungi API LLM: {e}")
            return [] # Kembalikan list kosong jika gagal
        except (json.JSONDecodeError, IndexError) as e:
            print(f"❌ Gagal mem-parse respons dari LLM: {e}")
            return []
        
    def _calculate_all_archetype_characteristics(self, df_segmentation, df_genre_map):
        all_archetype_profiles = []
        # Balikkan peta genre menjadi kode -> nama
        genre_code_to_name = pd.Series(df_genre_map.genre.values, index=df_genre_map.GenreCode).to_dict()

        for archetype_id in sorted(df_segmentation['dominant_archetype'].unique()):
            df_archetype = df_segmentation[df_segmentation['dominant_archetype'] == archetype_id]
            if df_archetype.empty: continue
        try:
            avg_games = df_archetype['Total_Games'].mean()
            avg_playtime = df_archetype['Avg_Playtime'].mean()
            total_achievements = df_archetype['Total_Achievements'].sum()
            dominant_topic_id = df_archetype['Dominant_Topic_User'].mode()[0]
            
            topic_info_list = topic_modeling_repository.getTopicByClusterId(int(dominant_topic_id))
            print(topic_info_list)
            dominant_topic_obj = {"id": int(dominant_topic_id), "keywords": []}
            print(dominant_topic_obj)
            # Periksa apakah list tidak kosong
            if topic_info_list:
                topic_info = topic_info_list[0] 
                if topic_info and hasattr(topic_info, 'Keyword') and topic_info.Keyword:
                    dominant_topic_obj["keywords"] = [kw.strip() for kw in topic_info.Keyword.split(',')]


            if isinstance(df_archetype['Top_3_Genres'].iloc[0], str):
                top_3_genres_lists = df_archetype['Top_3_Genres'].apply(ast.literal_eval)
            else:
                top_3_genres_lists = df_archetype['Top_3_Genres']

            all_genres = [g for sublist in top_3_genres_lists for g in sublist]
            top_3_genre_codes = [code for code, count in Counter(all_genres).most_common(3)]
            top_3_genre_names = [genre_code_to_name.get(code, "Unknown") for code in top_3_genre_codes]

            profile = {
                "archetype_id": int(archetype_id),
                "average_game_owned": round(avg_games, 2),
                "average_playtime": round(avg_playtime, 2),
                "average_achievement": int(total_achievements / len(df_archetype)), # Rata-rata achievement
                "dominant_topic": dominant_topic_obj,
                "top_3_genres": top_3_genre_names
            }
            all_archetype_profiles.append(profile)
        except Exception as e:
            print(f"❌ Gagal memproses archetype {archetype_id}: {e}")
        print(f"✅ Berhasil menghitung karakteristik untuk {len(all_archetype_profiles)} arketipe.")        
        return all_archetype_profiles

    def run_segmentation_pipeline(self, steam_ids, user_id, steam_proses_obj, unique_process_dir):
        print("--- [PIPELINE SEGMENTASI DIMULAI] ---")
        try:
            # tidak perlu base_dir
            dominant_output_path = os.path.join(unique_process_dir, "dominant_topic_per_game.csv")
            proses_id = steam_proses_obj.Proses_id
           

            # ======================================================================
            # LANGKAH 1 & 2: MEMUAT, MENGGABUNGKAN, DAN MEMBERSIHKAN DATA
            # ======================================================================
            print("\n[1-2/6] Memuat, menggabungkan, dan membersihkan data...")
            # --- KODE BARU YANG SUDAH DIPERBAIKI ---
            player_data = []
            for steam_id in steam_ids:
                player_data_raw = player_game_data_repository.get_data_by_steam_id(steam_id)
                player_data_dict = queryResultToDict(player_data_raw)
                if player_data_dict:
                    print(f"Data pemain untuk {steam_id} ditemukan.")
                    player_data.extend(player_data_dict)
                else:
                    print(f"Melakukan Scrapping {steam_id}")
                    scrapping = get_steam_id_data(steam_id)
                    player_game_data_repository.bulk_create_from_list(scrapping)
                    player_data.extend(scrapping)

            df_players = pd.DataFrame(player_data)
            print(df_players)
            # Pastikan tipe data 'steam_id' adalah string setelah membuat DataFrame
            if 'steam_id' in df_players.columns:
                df_players['steam_id'] = df_players['steam_id'].astype(str)
            df_dominant_topics = pd.read_csv(dominant_output_path)
            print(df_dominant_topics)

            df_players.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').lower() for col in df_players.columns]
            df_dominant_topics.columns = [col.replace(' ', '_').lower() for col in df_dominant_topics.columns]
            
            df_players['game_name'] = df_players['game_name'].str.strip().str.lower()
            df_dominant_topics['game'] = df_dominant_topics['game'].str.strip().str.lower()
            df_merged = pd.merge(
                df_players, 
                df_dominant_topics[['game', 'dominant_topic']], 
                left_on='game_name',  # Gunakan 'game_name' dari df_players
                right_on='game',      # Gunakan 'game' dari df_dominant_topics
                how='inner'
            )
            print(df_merged)
            df_cleaned = df_merged.dropna(subset=['steam_id', 'game_name', 'playtime_hours', 'genres', 'achievements', 'dominant_topic'])
            df_cleaned = df_cleaned[df_cleaned['playtime_hours'] >= 0.167]
            df_cleaned.drop_duplicates(subset=['steam_id', 'game_name'], inplace=True)

            game_counts_per_user = df_cleaned.groupby('steam_id')['game_name'].nunique()
            q1_threshold = game_counts_per_user.quantile(0.25)
            active_steam_ids = game_counts_per_user[game_counts_per_user >= q1_threshold].index
            df_final = df_cleaned[df_cleaned['steam_id'].isin(active_steam_ids)].copy()

            if df_merged.empty:
                print("❌ Peringatan: Tidak ada game dari pemain yang cocok dengan data topik yang ada.")
                return self.failedOrSuccessRequest('failed', 404, {"message": "Tidak ada game dari pemain yang cocok dengan data topik yang ada. Proses tidak bisa dilanjutkan."})
            # ======================================================================
            # LANGKAH 3: PEMBUATAN PROFIL PEMAIN
            # ======================================================================
            print("\n[3/6] Membuat profil pemain...")
            all_genres_flat = [genre.strip() for genres in df_final['genres'] for genre in str(genres).split(',') if genre.strip()]
            unique_genres = sorted(list(set(all_genres_flat)))
            genre_to_code = {genre: i for i, genre in enumerate(unique_genres)}
            # Buat DataFrame dari peta ini dan simpan ke CSV
            genre_map_df = pd.DataFrame(list(genre_to_code.items()), columns=['genre', 'GenreCode'])
            summary_stats = df_final.groupby('steam_id').agg(Total_Games=('game_name', 'nunique'), Avg_Playtime=('playtime_hours', 'mean'), Total_Achievements=('achievements', 'sum'))
            def get_top_genres(series):
                all_genres = [genre.strip() for genres in series for genre in str(genres).split(',') if genre.strip()]
                return [genre for genre, count in Counter(all_genres).most_common(3)] if all_genres else []
            top_genres_per_user = df_final.groupby('steam_id')['genres'].apply(get_top_genres).rename('Top_3_Genres')
            dominant_topic_per_user = df_final.groupby('steam_id')['dominant_topic'].agg(lambda x: x.mode()[0]).rename('Dominant_Topic_User')
            df_profile = summary_stats.join(top_genres_per_user).join(dominant_topic_per_user).reset_index()
            # ======================================================================
            # LANGKAH 4: FEATURE ENGINEERING
            # ======================================================================
            print("\n[4/6] Membangun matriks fitur...")
            df_features = df_profile[['steam_id', 'Total_Games', 'Avg_Playtime', 'Total_Achievements']].set_index('steam_id')
            topic_proportions = df_final.groupby(['steam_id', 'dominant_topic']).size().unstack(fill_value=0)
            topic_proportions = topic_proportions.div(topic_proportions.sum(axis=1), axis=0).add_prefix('topic_prop_')
            df_features = df_features.join(topic_proportions, how='left').fillna(0)

            df_genres_exploded = df_final[['steam_id', 'genres']].copy()
            df_genres_exploded['genres'] = df_genres_exploded['genres'].str.split(',')
            df_genres_exploded = df_genres_exploded.explode('genres')
            df_genres_exploded['genres'] = df_genres_exploded['genres'].str.strip()
            genre_proportions = df_genres_exploded.groupby(['steam_id', 'genres']).size().unstack(fill_value=0)
            genre_proportions = genre_proportions.div(genre_proportions.sum(axis=1), axis=0).add_prefix('genre_prop_')
            df_features = df_features.join(genre_proportions, how='left').fillna(0)
            
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(df_features)
            df_features_scaled = pd.DataFrame(features_scaled, index=df_features.index, columns=df_features.columns)
            # df_features_scaled.to_csv(features_output_path)

            # ======================================================================
            # LANGKAH 5: ANALISIS ARKETIPE (AA)
            # ======================================================================
            print("\n[5/6] Menjalankan Analisis Arketipe...")
            X = df_features_scaled.values
            def compute_archetypes(X, K, max_iter=100, tol=1e-4):
                n, d = X.shape
                rng = np.random.RandomState(42)
                Z = X[rng.choice(n, K, replace=(K > n))].copy()
                for _ in range(max_iter):
                    A = np.zeros((n, K))
                    for i in range(n): A[i], _ = nnls(Z.T, X[i])
                    Z_new = np.zeros_like(Z)
                    for j in range(K):
                        w = A[:, j][:, None]
                        Z_new[j] = (w * X).sum(axis=0) / w.sum() if w.sum() > 0 else Z[j]
                    if np.linalg.norm(Z - Z_new) < tol: break
                    Z = Z_new
                return A / (A.sum(axis=1, keepdims=True) + 1e-9), Z

            Ks = range(2, 11)
            errors = [np.linalg.norm(X - compute_archetypes(X, K)[0].dot(compute_archetypes(X, K)[1]), 'fro') for K in Ks]
            if np.allclose(errors, errors[0]):
                # Semua error sama → data terlalu mirip / jumlah fitur terlalu kecil
                best_K = 2  # default
            else:
                kl = KneeLocator(Ks, errors, curve='convex', direction='decreasing')
                best_K = kl.elbow or Ks[np.argmin(errors)]
            print(f"-> Jumlah segmen (K) optimal yang ditemukan: {best_K}")
            A_final, Z_final = compute_archetypes(X, best_K)

            # ======================================================================
            # LANGKAH 6: MENYIMPAN DAN VISUALISASI HASIL
            # ======================================================================
            players_data_path = os.path.join(unique_process_dir, "players_data.csv")
            genre_map_path = os.path.join(unique_process_dir, "hasil_peta_genre.csv")
            segmentation_csv_path = os.path.join(unique_process_dir, "hasil_segmentasi_pengguna.csv")
            membership_csv_path = os.path.join(unique_process_dir, "hasil_keanggotaan_arketipe.csv")
            karakteristik_json_path = os.path.join(unique_process_dir, 'karakteristik_arketipe_enriched.json')
            interpretasi_json_path = os.path.join(unique_process_dir, 'interpretasi_arketipe.json')


            print("\n[6/6] Menyimpan dan visualisasi hasil...")
            df_segmentation_result = df_profile.set_index('steam_id')
            for k in range(best_K): df_segmentation_result[f'archetype_{k+1}_weight'] = A_final[:, k]
            df_segmentation_result['dominant_archetype'] = np.argmax(A_final, axis=1) + 1

            dominant_archetypes = np.argmax(A_final, axis=1) + 1
            steam_ids_list = df_features_scaled.index.tolist()
            membership = {f"Arketipe_{k+1}": [] for k in range(best_K)}
            for sid, arc_num in zip(steam_ids_list, dominant_archetypes): membership[f"Arketipe_{arc_num}"].append(sid)
            max_len = max(len(v) for v in membership.values()); [v.extend([np.nan] * (max_len - len(v))) for v in membership.values()]

            karakteristik = self._calculate_all_archetype_characteristics(df_segmentation_result.reset_index(), genre_map_df)
            interpretasi = self._generate_archetype_interpretations(karakteristik, proses_id)

            df_players.to_csv(players_data_path, index=False)
            pd.DataFrame(membership).to_csv(membership_csv_path, index=False)
            genre_map_df.to_csv(genre_map_path, index=False)
            df_segmentation_result.to_csv(segmentation_csv_path)
            with open(karakteristik_json_path, 'w', encoding='utf-8') as f:
                json.dump(karakteristik, f, ensure_ascii=False, indent=4)
            with open(interpretasi_json_path, 'w', encoding='utf-8') as f:
                json.dump(interpretasi, f, ensure_ascii=False, indent=4)

            # save
            # 6e. Siapkan data untuk disimpan ke DB
            summary_data_to_save = {
                "Proses_id": proses_id,
                "segmentation_csv_path": segmentation_csv_path,
                "membership_csv_path": membership_csv_path,
                "karakteristik_json_path": karakteristik_json_path, 
                "interpretasi_json_path": interpretasi_json_path 
            }

            segmentasiRepository.create_summary(summary_data_to_save)
            print("\n--- [PIPELINE SEGMENTASI SELESAI] ---")
            return self.failedOrSuccessRequest('success', 200, {
                "message": "Proses segmentasi berhasil dan ringkasan telah disimpan.",
                "proses_id": proses_id,
                "karakteristik": karakteristik,
                "interpretasi": interpretasi
            })
        except FileNotFoundError as e:
            error_message = f"Failed: File not found - {e.filename}"
            print(f"\n❌ {error_message}")
            return error_message
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"Failed: An unexpected error occurred - {str(e)}"
            print(f"\n❌ {error_message}")
            return error_message