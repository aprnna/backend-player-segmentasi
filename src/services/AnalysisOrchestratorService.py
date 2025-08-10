
from src.utils.errorHandler import errorHandler
from src.utils.convert import queryResultToDict
from  src.services.Service import Service
from src.services.SegmentasiService import SegmentationService
from src.services.TopicModelingService import TopicModelingService
from src.repositories.SteamIDProsesRepository import SteamIDProsesRepository

topic_modeling_service = TopicModelingService()
segmentation_service = SegmentationService()
steam_id_proses_repository = SteamIDProsesRepository()

class AnalysisOrchestratorService(Service):
    @staticmethod
    def failedOrSuccessRequest(status, code, data):
        return {'status': status, "code": code, 'data': data}
    def run_full_analysis_pipeline(self, steam_ids, user_id):
        """
        Menjalankan seluruh pipeline analisis dari Topic Modeling hingga Segmentasi
        untuk satu ID Proses.
        """
        proses_catatan = None
        try:
            # ==========================================================
            # LANGKAH 1: BUAT CATATAN PROSES BARU
            # ==========================================================
            print(f"--- [PIPELINE GABUNGAN DIMULAI] ---")
            print(f"📝 Mencatat proses baru untuk User ID: {user_id} dengan Steam IDs: {steam_ids}")
            steam_ids_str = ", ".join(steam_ids)
            proses_catatan = steam_id_proses_repository.createNewSteamIDProses(steam_ids=steam_ids_str, user_id=user_id)
            proses_id = proses_catatan.Proses_id
            print(f"✅ Proses berhasil dicatat dengan ID: {proses_id}")

            # ==========================================================
            # LANGKAH 2: JALANKAN PROSES TOPIC MODELING
            # ==========================================================
            print("\n--- [MEMULAI SUB-PROSES: TOPIC MODELING] ---")
            # Panggil logika inti dari TopicModelingService
            # Catatan: Kita refaktor sedikit agar bisa dipanggil dari sini
            topic_modeling_result = topic_modeling_service.createNewTopicModeling(
                steam_ids=steam_ids, 
                userId=user_id, 
                steam_proses_obj=proses_catatan # Kirim objek proses yang sudah dibuat
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
            segmentation_result = segmentation_service.run_segmentation_pipeline(
                steam_ids=steam_ids, 
                user_id=user_id, 
                steam_proses_obj=proses_catatan # Kirim objek proses yang sama
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
            error_message = f"Terjadi kesalahan pada pipeline gabungan: {str(e)}"
            print(f"❌ {error_message}")
            return self.failedOrSuccessRequest('failed', 500, {"message": error_message})