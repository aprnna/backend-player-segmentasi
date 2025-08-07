import requests
import pandas as pd
import time
import os
import glob
from tqdm import tqdm

# Ganti dengan Steam API Key Anda
API_KEY = 'CF79089808D1F5AE17D9B3B5441E2DB8'
# Fungsi untuk mengambil nama game berdasarkan AppID
def get_game_name(appid):
    print(f"📌 Mengambil nama game untuk AppID: {appid}")
    url = f'https://store.steampowered.com/api/appdetails?appids={appid}&cc=us'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Gagal mengambil nama game untuk AppID: {appid}")
        return 'Unknown'

    try:
        data = response.json()
        if data.get(str(appid), {}).get('success', False):
            return data[str(appid)]['data'].get('name', 'Unknown')
    except ValueError:
        return 'Unknown'

    return 'Unknown'

# Fungsi untuk mendapatkan genre game
def get_game_genre(appid):
    print(f"📌 Mengambil genre untuk AppID: {appid}")
    url = f'https://store.steampowered.com/api/appdetails?appids={appid}&cc=us'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Gagal mengambil genre untuk AppID: {appid}")
        return []

    try:
        data = response.json()
        if data.get(str(appid), {}).get('success', False):
            genres = data[str(appid)]['data'].get('genres', [])
            return [genre['description'] for genre in genres]
    except ValueError:
        return []

    return []

# Fungsi untuk mengambil jumlah achievement yang sudah dicapai pemain
def get_game_achievements(steam_id, appid):
    print(f"🏆 Mengambil achievement untuk Steam ID: {steam_id}, AppID: {appid}")
    url = f'https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v1/?key={API_KEY}&steamid={steam_id}&appid={appid}'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Gagal mengambil achievement untuk Steam ID: {steam_id}, AppID: {appid}")
        return 0

    try:
        data = response.json()
        if 'playerstats' in data and 'achievements' in data['playerstats']:
            return sum(1 for ach in data['playerstats']['achievements'] if ach.get('achieved') == 1)
    except ValueError:
        return 0

    return 0

# Fungsi untuk mengambil review game
def get_game_reviews(appid, num_reviews=5):
    print(f"📝 Mengambil review untuk AppID: {appid}")
    url = f'https://store.steampowered.com/appreviews/{appid}?json=1&language=all'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Gagal mengambil review untuk AppID: {appid}")
        return []

    try:
        data = response.json()
        return [review['review'] for review in data.get('reviews', [])[:num_reviews]]
    except ValueError:
        return []

# Fungsi untuk mengambil daftar game dari Steam ID
def get_owned_games_by_steam_id(steam_id, index, total):
    print(f"\n🔄 [{index+1}/{total}] Mengambil daftar game untuk Steam ID: {steam_id}")
    url = f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&format=json'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Gagal mengambil daftar game untuk Steam ID: {steam_id}")
        return []

    try:
        data = response.json()
        games_data = []
        games = data.get('response', {}).get('games', [])
        print(f"✅ Steam ID {steam_id} memiliki {len(games)} game.")

        for i, game in enumerate(games):
            appid = game['appid']
            playtime = game['playtime_forever'] / 60
            genres = get_game_genre(appid)
            game_name = get_game_name(appid)
            achievement_count = get_game_achievements(steam_id, appid)
            reviews = get_game_reviews(appid, num_reviews=5)
            reviews_text = ' | '.join(reviews) if reviews else 'No Reviews'

            games_data.append({
                'Steam ID': steam_id,
                'App ID': appid,
                'Game Name': game_name,
                'Playtime (hours)': playtime,
                'Genres': ', '.join(genres),
                'Achievements': achievement_count,
                'Reviews': reviews_text
            })

            print(f"   🕹️ [{i+1}/{len(games)}] {game_name} berhasil diproses! 🎮")

            time.sleep(2)  # Hindari rate limiting

            # SAMPLE: Hentikan jika sudah cukup data
            # Misalnya, jika kita hanya butuh 3 game per Steam ID
            if len(games_data) >= 3:
                print("✅ Sudah cukup review yang diambil, menghentikan proses.")
                break
            
            
        return games_data
    except ValueError:
        return []


def get_reviews_from_steam_ids(steam_ids: list):
    all_reviews = []
    try:
        for idx, steam_id in enumerate(steam_ids):
          print(f"\n🔄 [{idx+1}/{len(steam_ids)}] Memproses Steam ID: {steam_id}")
          # Pastikan steam_id adalah string
          if not isinstance(steam_id, str):
              print(f"⚠️ Steam ID {steam_id} tidak valid, harus berupa string.")
              continue
          # Ambil daftar game untuk Steam ID ini
          games_data = get_owned_games_by_steam_id(steam_id, idx, len(steam_ids))
          if not games_data:
              print(f"⚠️ Tidak ada game ditemukan untuk Steam ID: {steam_id}")
              continue
          all_reviews.extend(games_data)
          print(f"✅ Selesai memproses Steam ID: {steam_id}")
          time.sleep(1)
         
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"⚠️ Gagal memproses Steam ID {steam_id}: {e}")

    return all_reviews