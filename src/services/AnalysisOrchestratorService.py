
from importlib.metadata import files
from src.utils.errorHandler import errorHandler
from src.utils.convert import queryResultToDict
from  src.services.Service import Service
from src.services.SegmentasiService import SegmentationService
from src.services.TopicModelingService import TopicModelingService
from src.repositories.SteamIDProsesRepository import SteamIDProsesRepository
import json
import pandas as pd
import os
from src.utils.uploadFIle import upload_file, delete_file
from src.utils.job_manager import job_status
from src.server.main import db

topic_modeling_service = TopicModelingService()
segmentation_service = SegmentationService()
steam_id_proses_repository = SteamIDProsesRepository()

class AnalysisOrchestratorService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {'status': status, "code": code, 'data': data}
    def getAllProsesByUserId(self, user_id):
        data = steam_id_proses_repository.getAllProsesByUserId(user_id)
        return self.failedOrSuccessRequest('success', 200, queryResultToDict(data))
    def getProsesById(self, proses_id):
        data = steam_id_proses_repository.getSteamIDProsesByProsesId(proses_id)
        return self.failedOrSuccessRequest('success', 200, queryResultToDict([data])[0])
    def getDetailProses(self, proses_id, user_id):
        try:
            proses_data = steam_id_proses_repository.getDetailProses(proses_id, user_id)
            if not proses_data:
                return self.failedOrSuccessRequest('failed', 404, {"message": "Proses tidak ditemukan atau Anda tidak memiliki akses."})
            if not proses_data.segmentasi_entries:
                return self.failedOrSuccessRequest('failed', 404, {"message": "Data ringkasan segmentasi untuk proses ini tidak ditemukan."})
            
            summary_record = proses_data.segmentasi_entries[0]
            try:
                with open(summary_record.karakteristik_json_path, 'r', encoding='utf-8') as f:
                    karakteristik = json.load(f)
                with open(summary_record.interpretasi_json_path, 'r', encoding='utf-8') as f:
                    interpretasi = json.load(f)
            except FileNotFoundError:
                return self.failedOrSuccessRequest('failed', 500, {"message": "File hasil analisis tidak ditemukan di server."})
            
            membership_rows = []
            try:
                membership_df = pd.read_csv(summary_record.membership_csv_path)
                # 1. Ubah nilai kosong (NaN) menjadi string kosong
                membership_df.fillna('', inplace=True)
                # 3. Tambahkan ID unik untuk setiap baris (diperlukan oleh React/Next.js)
                membership_df.reset_index(inplace=True)
                membership_df = membership_df.rename(columns={'index': 'id'})
                # 4. Konversi DataFrame menjadi array of objects
                membership_rows = membership_df.to_dict(orient='records')
                
            except FileNotFoundError:
                print(f"⚠️ Peringatan: File keanggotaan {summary_record.membership_csv_path} tidak ditemukan.")
            except Exception as e:
                print(f"⚠️ Peringatan: Gagal memproses file keanggotaan: {e}")
            # 4. Susun respons akhir
            result = {
                "Proses_id": proses_data.Proses_id,
                "Steam_ids": proses_data.Steam_id.split(','), # Ubah string menjadi list
                "User_id": proses_data.User_id,
                "segmentasi_summary": {
                    "karakteristik_arketipe": karakteristik,
                    "interpretasi_arketipe": interpretasi,
                    "membership_data": membership_rows,
                    "file_referensi": {
                        "hasil_lengkap_csv": summary_record.segmentation_csv_path,
                        "daftar_anggota_csv": summary_record.membership_csv_path
                    }
                },
                "topics": [
                    {
                        "Topic_id": t.Topic_id,
                        "Keyword": t.Keyword,
                        "Cluster": t.Cluster
                    }
                    for t in proses_data.topics
                ]
            }
            return self.failedOrSuccessRequest('success', 200, result) 
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.failedOrSuccessRequest('failed', 500, {"message": f"Terjadi kesalahan internal: {str(e)}"})
    

    def run_full_analysis_pipeline(self, steam_ids, files,user_id, job_id):
        """
        Menjalankan seluruh pipeline analisis dari Topic Modeling hingga Segmentasi
        untuk satu ID Proses.
        """
        # cek apkah ada file jika ada ambil semua data colom steam_id dan append ke list steam_ids
        # Perbaikan: Cek apakah ada file dengan nama 'file' di request.files
        job_status[job_id].update({
                'status': 'processing',
                'progress': 15,
                'message': "loading data...."
            })
            
        if "file" in files and files["file"] is not None:
            csv_path = files["file"]
            try:
                df = pd.read_csv(csv_path)
                print(f"✅ Loaded file, shape={df.shape}")
                print(df)
                if df.empty:
                    delete_file(csv_path)
                    return self.failedOrSuccessRequest("failed", 400, "File CSV kosong")

                if "steamid" not in df.columns:
                    delete_file(csv_path)
                    return self.failedOrSuccessRequest("failed", 400, "File CSV harus memiliki kolom 'steamid'")

                csv_steam_ids = df["steamid"].dropna().astype(str).tolist()
                if not csv_steam_ids:
                    delete_file(csv_path)
                    return self.failedOrSuccessRequest("failed", 400, "Kolom steamid kosong")

                steam_ids.extend(csv_steam_ids)
                steam_ids = list(set(steam_ids))  # hapus duplikat
            except Exception as e:
                return self.failedOrSuccessRequest("failed", 500, f"Error membaca file CSV: {e}")

        if not steam_ids:
            return self.failedOrSuccessRequest("failed", 400, "Tidak ada Steam ID yang valid")
        job_status[job_id].update({
            'status': 'processing',
            'progress': 20,
            'message': f'Success load {len(steam_ids)} Steam IDs'
        })
        print(f"🎮 Total Steam IDs untuk dianalisis: {len(steam_ids)}")
        proses_catatan = None
        try:
            # ==========================================================
            # LANGKAH 1: BUAT CATATAN PROSES BARU
            # ==========================================================
            print(f"--- [PIPELINE GABUNGAN DIMULAI] ---")
            print(f"📝 Mencatat proses baru untuk User ID: {user_id} dengan Steam IDs: {steam_ids}")
            steam_ids_str = ", ".join(steam_ids)
            proses_catatan = steam_id_proses_repository.createNewSteamIDProses(steam_ids=steam_ids_str, user_id=user_id)
            print(proses_catatan)
            proses_id = proses_catatan.Proses_id
            base_output_dir = "public/generated_outputs"
            print(f"✅ Proses berhasil dicatat dengan ID: {proses_id}")
            unique_process_dir = os.path.join(base_output_dir, f"proses_{proses_id}")
            # Buat folder jika belum ada
            os.makedirs(unique_process_dir, exist_ok=True)
            print(f"🗂️ Semua file output untuk Proses ID {proses_id} akan disimpan di: {unique_process_dir}")
            # ==========================================================
            # LANGKAH 2: JALANKAN PROSES TOPIC MODELING
            # ==========================================================
            print("\n--- [MEMULAI SUB-PROSES: TOPIC MODELING] ---")
            # Panggil logika inti dari TopicModelingService
            # Catatan: Kita refaktor sedikit agar bisa dipanggil dari sini
            job_status[job_id].update({
                'status': 'processing',
                'progress': 25,
                'message': 'Running Topic Modeling...'
            })
            topic_modeling_result = topic_modeling_service.createNewTopicModeling(
                steam_ids=steam_ids, 
                userId=user_id, 
                steam_proses_obj=proses_catatan,
                unique_process_dir=unique_process_dir,
                job_id=job_id
            )

            if topic_modeling_result.get('status') == 'failed':
                # Jika topic modeling gagal, hentikan proses
                raise Exception(f"Sub-proses Topic Modeling gagal: {topic_modeling_result.get('data', {}).get('message', 'Error tidak diketahui')}")
            
            print("✅ Sub-proses Topic Modeling selesai.")
            
            # ==========================================================
            # LANGKAH 3: JALANKAN PROSES SEGMENTASI
            # ==========================================================
            print("\n--- [MEMULAI SUB-PROSES: SEGMENTASI] ---")
            # Panggil logika inti dari SegmentationService
            job_status[job_id].update({
                'status': 'processing',
                'progress': 70,
                'message': 'Running Segmentasi...'
            })
            segmentation_result = segmentation_service.run_segmentation_pipeline(
                steam_ids=steam_ids, 
                user_id=user_id, 
                steam_proses_obj=proses_catatan,
                unique_process_dir=unique_process_dir,
                job_id=job_id
            )
            if segmentation_result.get('status') == 'failed':
                # Jika segmentasi gagal, hentikan proses
                raise Exception(f"Sub-proses Segmentasi gagal: {segmentation_result.get('data', {}).get('message', 'Error tidak diketahui')}")

            print("✅ Sub-proses Segmentasi selesai.")
            
            # ==========================================================
            # LANGKAH 4: SELESAI
            # ==========================================================
            print("\n--- [PIPELINE GABUNGAN SELESAI] ---")
            return self.failedOrSuccessRequest('success', 200, {
                "message": "Proses analisis lengkap dari topic modeling hingga segmentasi berhasil.",
                "proses_id": proses_id,
                "topic_modeling_summary": topic_modeling_result.get('data'),
                "segmentation_summary": segmentation_result.get('data')
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            db.session.rollback()
            error_message = f"Terjadi kesalahan pada pipeline gabungan: {str(e)}"
            print(f"❌ {error_message}")
            return self.failedOrSuccessRequest('failed', 500, {"message": error_message})