# Di dalam src/repositories/PlayerGameDataRepository.py

from src.models.PlayerGameData import PlayerGameData, db

class PlayerGameDataRepository:
    def bulk_create_from_list(self, player_data_list):
        """
        Menyimpan list of dictionary data scraping ke database secara massal
        dengan Proses_id yang sama.
        """
        if not player_data_list:
            return True # Tidak ada yang perlu disimpan

        try:
            new_entries = []
            for record in player_data_list:
                new_entry = PlayerGameData(
                    steam_id=record.get('steam_id'),
                    app_id=record.get('app_id'),
                    game_name=record.get('game_name'),
                    playtime_hours=record.get('playtime_hours'),
                    genres=record.get('genres'),
                    achievements=record.get('achievements')
                )
                new_entries.append(new_entry)
            
            db.session.bulk_save_objects(new_entries)
            db.session.commit()
            print(f"✅ Berhasil menyimpan {len(new_entries)}.")
            return True

        except Exception as e:
            db.session.rollback()
            print(f"❌ Gagal melakukan bulk insert data game pemain: {e}")
            raise e

    def get_data_by_steam_id(self, steam_id):
        return PlayerGameData.query.filter_by(steam_id=steam_id).all()