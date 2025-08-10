from src.models.Segmentasi import Segmentasi, db

class SegmentasiRepository:
    def getAllSegmentasi(self):
        return Segmentasi.query.all()

    def getSegmentasiByProsesId(self, proses_id):
        return Segmentasi.query.filter_by(Proses_id=proses_id).first()

    def createNewSegmentasi(self, data):
        newSegmentasi = Segmentasi(
            steam_id=data['steam_id'],
            total_games=data['total_games'],
            avg_playtime=data['avg_playtime'],
            total_achievements=data['total_achievements'],
            top_3_genres=data['top_3_genres'],
            dominant_topic_user=data['dominant_topic_user'],
            dominant_archetype=data['dominant_archetype'],
            archetype_weights=data['archetype_weights']
        )
        db.session.add(newSegmentasi)
        db.session.commit()
        return newSegmentasi

    # Modifikasi metode bulk create agar menerima Proses_id
    def bulk_create_from_dataframe(self, df_segmentation_result, proses_id):
        """
        Menyimpan data dari DataFrame ke database dengan Proses_id yang sama.
        """
        try:
            new_segmentations = []
            weight_columns = [col for col in df_segmentation_result.columns if 'archetype_' in col and '_weight' in col]

            for steam_id, row in df_segmentation_result.iterrows():
                archetype_weights_dict = {
                    # Key menjadi 'archetype_1', 'archetype_2', dst.
                    col.replace('_weight', ''): row[col] for col in weight_columns
                }
                
                new_entry = Segmentasi(
                    proses_id=proses_id,  # Link ke proses yang sedang berjalan
                    steam_id=steam_id,
                    total_games=int(row.get('Total_Games')),
                    avg_playtime=float(row.get('Avg_Playtime')),
                    total_achievements=int(row.get('Total_Achievements')),
                    top_3_genres=str(row.get('Top_3_Genres')),
                    dominant_topic_user=int(row.get('Dominant_Topic_User')),
                    dominant_archetype=int(row.get('dominant_archetype')),
                    archetype_weights=archetype_weights_dict
                )
                new_segmentations.append(new_entry)
            
            db.session.bulk_save_objects(new_segmentations)
            db.session.commit()
            
            print(f"✅ Berhasil menyimpan {len(new_segmentations)} data segmentasi untuk Proses ID: {proses_id}.")
            return True
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Gagal melakukan bulk insert ke database: {e}")
            raise e