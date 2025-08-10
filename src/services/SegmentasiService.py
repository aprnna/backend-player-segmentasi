from  src.services.Service import Service
from src.utils.convert import queryResultToDict
from src.utils.errorHandler import errorHandler
from src.repositories.SegmentasiRepository import SegmentasiRepository
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

segmentasiRepository = SegmentasiRepository()

class SegmentationService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {
            'status': status,
            "code": code,
            'data': data,
        }
    def run_segmentation_pipeline(self, steam_ids, user_id, steam_proses_obj):
      print("--- [PIPELINE SEGMENTASI DIMULAI] ---")
      try:
          # tidak perlu base_dir
          dominant_topics_path = "dominant_topic_per_game.csv"

          profile_output_path = "hasil_profil_pemain.csv"
          features_output_path = "hasil_fitur_untuk_model.csv"
          segmentation_output_path = "hasil_segmentasi_pengguna.csv"
          membership_output_path = "hasil_keanggotaan_arketipe.csv"
          elbow_plot_path = "hasil_plot_elbow.png"
          umap_plot_path = "hasil_plot_segmentasi_umap.png"
          proses_id = steam_proses_obj.Proses_id

          # ======================================================================
          # LANGKAH 1 & 2: MEMUAT, MENGGABUNGKAN, DAN MEMBERSIHKAN DATA
          # ======================================================================
          print("\n[1-2/6] Memuat, menggabungkan, dan membersihkan data...")
          # --- KODE BARU YANG SUDAH DIPERBAIKI ---
          player_data_list = get_steam_id_data(steam_ids)
          if not player_data_list:
              return self.failedOrSuccessRequest('failed', 404, "Tidak ada data pemain yang berhasil di-scraping.")

          # Langsung ubah list dictionary menjadi DataFrame
          df_players = pd.DataFrame(player_data_list)
          df_players.to_csv('players_data.csv', index=False)
          # Pastikan tipe data 'Steam ID' adalah string setelah membuat DataFrame
          if 'Steam ID' in df_players.columns:
            df_players['Steam ID'] = df_players['Steam ID'].astype(str)
          df_dominant_topics = pd.read_csv(dominant_topics_path)
          print(df_dominant_topics)
          
          df_players['Game Name'] = df_players['Game Name'].str.strip().str.lower()
          df_dominant_topics['Game'] = df_dominant_topics['Game'].str.strip().str.lower()
          
          df_merged = pd.merge(df_players, df_dominant_topics[['Game', 'Dominant_Topic']], left_on='Game Name', right_on='Game', how='inner')
          
          df_cleaned = df_merged.dropna(subset=['Steam ID', 'Game Name', 'Playtime (hours)', 'Genres', 'Achievements', 'Dominant_Topic'])
          df_cleaned = df_cleaned[df_cleaned['Playtime (hours)'] >= 0.167]
          df_cleaned.drop_duplicates(subset=['Steam ID', 'Game Name'], inplace=True)
          
          game_counts_per_user = df_cleaned.groupby('Steam ID')['Game Name'].nunique()
          q1_threshold = game_counts_per_user.quantile(0.25)
          active_steam_ids = game_counts_per_user[game_counts_per_user >= q1_threshold].index
          df_final = df_cleaned[df_cleaned['Steam ID'].isin(active_steam_ids)].copy()

          if df_merged.empty:
              print("❌ Peringatan: Tidak ada game dari pemain yang cocok dengan data topik yang ada.")
              return self.failedOrSuccessRequest('failed', 404, {"message": "Tidak ada game dari pemain yang cocok dengan data topik yang ada. Proses tidak bisa dilanjutkan."})
          # ======================================================================
          # LANGKAH 3: PEMBUATAN PROFIL PEMAIN
          # ======================================================================
          print("\n[3/6] Membuat profil pemain...")
          summary_stats = df_final.groupby('Steam ID').agg(Total_Games=('Game Name', 'nunique'), Avg_Playtime=('Playtime (hours)', 'mean'), Total_Achievements=('Achievements', 'sum'))
          def get_top_genres(series):
              all_genres = [genre.strip() for genres in series for genre in str(genres).split(',') if genre.strip()]
              return [genre for genre, count in Counter(all_genres).most_common(3)] if all_genres else []
          top_genres_per_user = df_final.groupby('Steam ID')['Genres'].apply(get_top_genres).rename('Top_3_Genres')
          dominant_topic_per_user = df_final.groupby('Steam ID')['Dominant_Topic'].agg(lambda x: x.mode()[0]).rename('Dominant_Topic_User')
          df_profile = summary_stats.join(top_genres_per_user).join(dominant_topic_per_user).reset_index()
          df_profile.to_csv(profile_output_path, index=False)
          
          # ======================================================================
          # LANGKAH 4: FEATURE ENGINEERING
          # ======================================================================
          print("\n[4/6] Membangun matriks fitur...")
          df_features = df_profile[['Steam ID', 'Total_Games', 'Avg_Playtime', 'Total_Achievements']].set_index('Steam ID')
          topic_proportions = df_final.groupby(['Steam ID', 'Dominant_Topic']).size().unstack(fill_value=0)
          topic_proportions = topic_proportions.div(topic_proportions.sum(axis=1), axis=0).add_prefix('topic_prop_')
          df_features = df_features.join(topic_proportions, how='left').fillna(0)
          
          df_genres_exploded = df_final[['Steam ID', 'Genres']].copy()
          df_genres_exploded['Genres'] = df_genres_exploded['Genres'].str.split(',')
          df_genres_exploded = df_genres_exploded.explode('Genres')
          df_genres_exploded['Genres'] = df_genres_exploded['Genres'].str.strip()
          genre_proportions = df_genres_exploded.groupby(['Steam ID', 'Genres']).size().unstack(fill_value=0)
          genre_proportions = genre_proportions.div(genre_proportions.sum(axis=1), axis=0).add_prefix('genre_prop_')
          df_features = df_features.join(genre_proportions, how='left').fillna(0)
          
          scaler = MinMaxScaler()
          features_scaled = scaler.fit_transform(df_features)
          df_features_scaled = pd.DataFrame(features_scaled, index=df_features.index, columns=df_features.columns)
          df_features_scaled.to_csv(features_output_path)

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
          kl = KneeLocator(Ks, errors, curve='convex', direction='decreasing')
          best_K = kl.elbow or Ks[np.argmin(errors)]
          print(f"-> Jumlah segmen (K) optimal yang ditemukan: {best_K}")
          # plt.figure(figsize=(8, 5)); plt.plot(Ks, errors, 'o-'); 
          # if kl.elbow: plt.axvline(kl.elbow, color='r', linestyle='--')
          # plt.title('Metode Elbow untuk Analisis Arketipe'); plt.xlabel('Jumlah Segmen (K)'); plt.ylabel('Error')
          # plt.savefig(elbow_plot_path)
          
          A_final, Z_final = compute_archetypes(X, best_K)

          # ======================================================================
          # LANGKAH 6: MENYIMPAN DAN VISUALISASI HASIL
          # ======================================================================
          print("\n[6/6] Menyimpan dan visualisasi hasil...")
          df_segmentation_result = df_profile.set_index('Steam ID')
          for k in range(best_K): df_segmentation_result[f'archetype_{k+1}_weight'] = A_final[:, k]
          df_segmentation_result['dominant_archetype'] = np.argmax(A_final, axis=1) + 1
          df_segmentation_result.to_csv(segmentation_output_path)

          dominant_archetypes = np.argmax(A_final, axis=1) + 1
          steam_ids_list = df_features_scaled.index.tolist()
          membership = {f"Arketipe_{k+1}": [] for k in range(best_K)}
          for sid, arc_num in zip(steam_ids_list, dominant_archetypes): membership[f"Arketipe_{arc_num}"].append(sid)
          max_len = max(len(v) for v in membership.values()); [v.extend([np.nan] * (max_len - len(v))) for v in membership.values()]
          pd.DataFrame(membership).to_csv(membership_output_path, index=False)

          # coords_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(A_final)
          # plt.figure(figsize=(10, 8))
          # scatter = plt.scatter(coords_umap[:, 0], coords_umap[:, 1], c=dominant_archetypes, cmap=plt.get_cmap("viridis", best_K), alpha=0.7)
          # cbar = plt.colorbar(scatter, ticks=range(1, best_K + 1)); cbar.set_label("Segmen Dominan")
          # plt.title("Visualisasi Segmen Pengguna"); plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
          # plt.savefig(umap_plot_path)
          segmentasiRepository.bulk_create_from_dataframe(df_segmentation_result, proses_id=proses_id)

          print("\n--- [PIPELINE SEGMENTASI SELESAI] ---")
          return self.failedOrSuccessRequest('success', 200, {
              "message": "Proses segmentasi berhasil diselesaikan dan disimpan ke database.",
              "proses_id": proses_id,
              "total_users_processed": len(df_segmentation_result)
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