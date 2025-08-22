import os
import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup
try:
    from fake_useragent import UserAgent
except Exception:
    UserAgent = None
import time
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Konfigurasi Folder
INPUT_DIR = 'Fixed'
OUTPUT_DIR = 'reviews_4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi Scraping
BASE_URL = "https://steamcommunity.com/profiles/"
MIN_DELAY = 3  # Delay minimum (detik)
MAX_DELAY = 7  # Delay maksimum (detik)
MAX_PAGES = 50  # Batas maksimal halaman per user
MAX_ATTEMPTS = 3  # Maksimal percobaan per halaman
# Paralel
MAX_WORKERS = 5  # Jumlah thread paralel (atur sesuai kemampuan / etika scraping)

# Semaphore untuk membatasi panggilan ke Store API secara bersamaan
STORE_API_SEMAPHORE = threading.Semaphore(2)  # batasi jumlah request ke store API simultan

# Inisialisasi UserAgent (dengan fallback)
try:
    UA = UserAgent() if UserAgent is not None else None
except Exception:
    UA = None

def get_random_delay():
    return random.uniform(MIN_DELAY, MAX_DELAY)

def _get_ua_string():
    if UA:
        try:
            return UA.random
        except Exception:
            return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    else:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"

def scrape_user_reviews(steam64_id):
    """
    Scrape semua review untuk satu steam64_id.
    Menggunakan requests.Session() per-call untuk efisiensi koneksi.
    """
    session = requests.Session()
    all_reviews = []
    page = 1

    while page <= MAX_PAGES:
        if page == 1:
            url = f"{BASE_URL}{steam64_id}/recommended/"
        else:
            url = f"{BASE_URL}{steam64_id}/recommended/?p={page}"

        attempt = 0
        success = False

        while attempt < MAX_ATTEMPTS and not success:
            try:
                delay = get_random_delay()
                time.sleep(delay)

                headers = {'User-Agent': _get_ua_string()}
                resp = session.get(url, headers=headers, timeout=15)

                if "Steam Community :: Error" in resp.text:
                    print(f"[!] Steam error page detected for {steam64_id}")
                    return all_reviews if all_reviews else [{"Error": "Steam Error Page", "UserID": steam64_id}]

                soup = BeautifulSoup(resp.text, 'html.parser')
                review_boxes = soup.find_all('div', class_='review_box')
                if not review_boxes:
                    if page == 1:
                        return [{"Status": "No reviews found", "UserID": steam64_id}]
                    return all_reviews

                for review in review_boxes:
                    try:
                        link_tag = review.find('div', class_='leftcol').find('a') if review.find('div', class_='leftcol') else None
                        game_url = link_tag['href'] if link_tag and link_tag.has_attr('href') else ""
                        game_name = "N/A"
                        genre_str = "N/A"

                        # Jika ada appid, panggil Store API (dengan semaphore)
                        m = re.search(r'/app/(\d+)', game_url)
                        if m:
                            appid = m.group(1)
                            api_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
                            try:
                                with STORE_API_SEMAPHORE:  # batasi panggilan ke API store
                                    api_resp = session.get(api_url, headers={'User-Agent': _get_ua_string()}, timeout=15)
                                    if api_resp.status_code == 200:
                                        data = api_resp.json()
                                        if data.get(appid, {}).get('success'):
                                            game_data = data[appid]['data']
                                            game_name = game_data.get('name', 'N/A')
                                            genres = game_data.get('genres', [])
                                            genre_list = [g.get('description','') for g in genres]
                                            genre_str = ', '.join([x for x in genre_list if x]) if genre_list else 'N/A'
                            except requests.exceptions.RequestException:
                                pass

                        rating_elem = review.find('div', class_='title')
                        content_elem = review.find('div', class_='content')
                        hours_elem = review.find('div', class_='hours')
                        date_elem = review.find('div', class_='postedDate')

                        all_reviews.append({
                            'Steam_ID': steam64_id,
                            'Page': page,
                            'Game': game_name,
                            'Genre': genre_str,
                            'Rating': "Recommended" if rating_elem and 'Recommended' in rating_elem.text else "Not Recommended",
                            'Review': content_elem.get_text(strip=True, separator=' ') if content_elem else "N/A",
                            'Playtime': hours_elem.get_text(strip=True) if hours_elem else "N/A",
                            'PostedDate': date_elem.get_text(strip=True) if date_elem else "N/A"
                        })
                    except Exception as e:
                        print(f"Error parsing review for {steam64_id}: {e}")
                        continue

                next_btn = soup.find('a', class_='pagebtn', string='>')
                if not next_btn:
                    return all_reviews

                success = True
                page += 1

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(f"Attempt {attempt} failed for page {page} of {steam64_id}: {e}")
                if attempt >= MAX_ATTEMPTS:
                    print(f"Max attempts reached for {steam64_id} page {page}")
                    return all_reviews if all_reviews else [{"Error": "Max attempts reached", "UserID": steam64_id}]
                time.sleep(delay * 2)

    print(f"Warning: Reached max page limit ({MAX_PAGES}) for {steam64_id}")
    return all_reviews

def process_csv_parallel(input_path, max_workers=MAX_WORKERS, save_csv=True):
    """
    Baca CSV dengan kolom 'Steam ID', lalu scrape paralel untuk semua Steam ID unik.
    Mengembalikan DataFrame hasil gabungan.
    """
    df = pd.read_csv(input_path)
    if 'Steam ID' not in df.columns:
        print(f"File {input_path} tidak memiliki kolom 'Steam ID'")
        return None

    steam_ids = [str(x).strip() for x in df['Steam ID'].dropna().unique()]
    # validasi singkat
    valid_ids = []
    for steam_id in steam_ids:
        if len(steam_id) == 17 and steam_id.isdigit():
            valid_ids.append(steam_id)
        else:
            print(f"ID tidak valid, dilewati: {steam_id}")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        future_to_id = {exc.submit(scrape_user_reviews, sid): sid for sid in valid_ids}
        for i, fut in enumerate(as_completed(future_to_id), 1):
            sid = future_to_id[fut]
            try:
                reviews = fut.result()
                if reviews:
                    results.extend(reviews)
                print(f"[{i}/{len(valid_ids)}] Selesai: {sid} -> {len(reviews)} item")
            except Exception as e:
                print(f"Error saat memproses {sid}: {e}")

    if not results:
        print("Tidak ada review yang dikumpulkan.")
        return None

    out_df = pd.DataFrame(results)
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(OUTPUT_DIR, f"combined_reviews_{timestamp}.csv")
        out_df.to_csv(fname, index=False)
        print(f"Hasil disimpan ke: {fname}")

    return out_df

def get_game_reviews_parallel(steam_ids: list, max_workers=MAX_WORKERS):
    """
    Versi paralel dari get_game_reviews — mengembalikan list of reviews.
    """
    # validasi input
    ids = [str(x).strip() for x in steam_ids if x is not None]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        future_to_id = {exc.submit(scrape_user_reviews, sid): sid for sid in ids}
        for fut in as_completed(future_to_id):
            sid = future_to_id[fut]
            try:
                raw_reviews = fut.result()
                results.extend(raw_reviews)
                print(f"Selesai: {sid} -> {len(raw_reviews)} item")
            except Exception as e:
                print(f"Error saat memproses {sid}: {e}")
    return results
