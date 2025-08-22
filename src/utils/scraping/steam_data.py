import requests
import pandas as pd
import time
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading

# CONFIG
API_KEY = 'CF79089808D1F5AE17D9B3B5441E2DB8'  # pertimbangkan menyimpan di env var untuk keamanan
OUTPUT_DIR = "reviews_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paralel / rate-limit settings
MAX_STEAMID_WORKERS = 5      # jumlah Steam ID yang diproses paralel
MAX_PER_GAME_WORKERS = 3     # jumlah task per-game paralel (untuk appdetails + achievements per steam id)
STORE_API_CONCURRENCY = 2    # berapa banyak request ke store.steampowered.com bersamaan
WEBAPI_CONCURRENCY = 2       # berapa banyak request ke api.steampowered.com bersamaan

# Semaphores untuk membatasi concurrent calls ke masing2 API
store_api_sem = threading.Semaphore(STORE_API_CONCURRENCY)
webapi_sem = threading.Semaphore(WEBAPI_CONCURRENCY)

# Helper: buat session dengan retry/backoff
def create_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115 Safari/537.36'})
    return s

# Ambil nama + genre (1 panggilan)
def get_game_details(session: requests.Session, appid: int, timeout=10):
    url = f'https://store.steampowered.com/api/appdetails?appids={appid}&cc=us'
    with store_api_sem:
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            app = data.get(str(appid))
            if app and app.get('success'):
                name = app['data'].get('name', 'Unknown')
                genres = [g.get('description', '') for g in app['data'].get('genres', [])]
                return name, ", ".join([g for g in genres if g])
        except Exception as e:
            # Jangan hentikan proses, kembalikan fallback
            # print(f"⚠️ get_game_details error for {appid}: {e}")
            return 'Unknown', ''
    return 'Unknown', ''

# Ambil jumlah achievement yang dicapai pemain untuk appid
def get_game_achievements(session: requests.Session, steam_id: str, appid: int, timeout=10):
    url = f'https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v1/?key={API_KEY}&steamid={steam_id}&appid={appid}'
    with webapi_sem:
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if 'playerstats' in data and 'achievements' in data['playerstats']:
                return sum(1 for ach in data['playerstats']['achievements'] if ach.get('achieved') == 1)
        except Exception as e:
            # print(f"⚠️ get_game_achievements error for {steam_id} app {appid}: {e}")
            return 0
    return 0

# Ambil daftar owned games untuk satu steam_id (synchronous HTTP call)
def fetch_owned_games(session: requests.Session, steam_id: str):
    url = f'https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={API_KEY}&steamid={steam_id}&format=json'
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        games = data.get('response', {}).get('games', [])
        return games
    except Exception as e:
        print(f"⚠️ Gagal mengambil daftar game untuk Steam ID {steam_id}: {e}")
        return []

# Proses semua game untuk satu Steam ID (paralel terbatas per-game)
def get_owned_games_by_steam_id_parallel(steam_id: str, per_game_workers=MAX_PER_GAME_WORKERS):
    session = create_session()
    games = fetch_owned_games(session, steam_id)
    if not games:
        # kosong atau error
        return []

    results = []
    # Batasi jumlah game yang diproses paralel agar tidak overload API
    with ThreadPoolExecutor(max_workers=min(per_game_workers, 8)) as ex:
        future_to_app = {}
        for game in games:
            appid = int(game['appid'])
            # submit a job that fetches details + achievements
            future = ex.submit(process_single_game, session, steam_id, appid, game)
            future_to_app[future] = appid

        # gunakan tqdm agar ada progress per steam id
        for fut in tqdm(as_completed(future_to_app), total=len(future_to_app),
                        desc=f"Processing games for {steam_id}", unit="game"):
            try:
                out = fut.result()
                if out:
                    results.append(out)
            except Exception as e:
                appid = future_to_app.get(fut, 'unknown')
                # print(f"⚠️ Error processing app {appid} for {steam_id}: {e}")
                continue

    return results

# Worker untuk setiap game: gabungkan get_game_details + get_game_achievements
def process_single_game(session: requests.Session, steam_id: str, appid: int, game_meta: dict):
    # playtime in hours
    playtime = game_meta.get('playtime_forever', 0) / 60.0
    # panggil details + achievements (dua panggilan yang dibatasi semaphore)
    name, genres = get_game_details(session, appid)
    achievements = get_game_achievements(session, steam_id, appid)
    return {
        'steam_id': steam_id,
        'app_id': appid,
        'game_name': name,
        'playtime_hours': playtime,
        'genres': genres,
        'achievements': achievements
    }

# Fungsi publik: terima list steam ids dan proses paralel per-steamid
def get_steam_ids_data_parallel(steam_ids: list, max_workers=MAX_STEAMID_WORKERS, save_csv=True, csv_name=None):
    # validasi steam_ids (pastikan string)
    steam_ids_clean = [str(s).strip() for s in steam_ids if s is not None]
    results_all = []

    # ThreadPool untuk Steam IDs
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_sid = {ex.submit(get_owned_games_by_steam_id_parallel, sid): sid for sid in steam_ids_clean}
        # gunakan tqdm untuk progress steam id
        for fut in tqdm(as_completed(future_to_sid), total=len(future_to_sid),
                        desc="Steam IDs", unit="steamid"):
            sid = future_to_sid[fut]
            try:
                sid_results = fut.result()
                if sid_results:
                    results_all.extend(sid_results)
                # beri jeda kecil antar ID agar tidak memicu rate limit
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Error processing Steam ID {sid}: {e}")

    if not results_all:
        print("⚠️ Tidak ada data game yang dikumpulkan.")
        return []

    # Convert ke DataFrame dan simpan CSV
    df = pd.DataFrame(results_all)
    if save_csv:
        if not csv_name:
            csv_name = f"owned_games_{len(steam_ids_clean)}ids.csv"
        out_path = os.path.join(OUTPUT_DIR, csv_name)
        df.to_csv(out_path, index=False)
        print(f"✅ Hasil disimpan ke {out_path}")

    return results_all

# Single steam id helper (return list hasil)
def get_steam_id_data(steam_id: str):
    return get_owned_games_by_steam_id_parallel(steam_id)
