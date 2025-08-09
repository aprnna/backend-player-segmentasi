import os
import glob
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import random
import re

# Konfigurasi Folder
INPUT_DIR = 'Fixed'
OUTPUT_DIR = 'reviews_4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Konfigurasi Scraping
BASE_URL = "https://steamcommunity.com/profiles/"
HEADERS = {'User-Agent': UserAgent().random}
MIN_DELAY = 3  # Delay minimum (detik)
MAX_DELAY = 7  # Delay maksimum (detik)
MAX_PAGES = 50  # Batas maksimal halaman
MAX_ATTEMPTS = 3  # Maksimal percobaan per halaman

def get_random_delay():
    return random.uniform(MIN_DELAY, MAX_DELAY)

def scrape_user_reviews(steam64_id):
    all_reviews = []
    page = 1

    while page <= MAX_PAGES:
        url = f"{BASE_URL}{steam64_id}/recommended/?p={page}" if page > 1 else f"{BASE_URL}{steam64_id}/recommended/"
        attempt = 0
        success = False

        while attempt < MAX_ATTEMPTS and not success:
            try:
                delay = get_random_delay()
                time.sleep(delay)
                headers = {'User-Agent': UserAgent().random}
                response = requests.get(url, headers=headers, timeout=15)

                if "Steam Community :: Error" in response.text:
                    print(f"[!] Steam error page detected for {steam64_id}")
                    return all_reviews if all_reviews else [{"Error": "Steam Error Page", "UserID": steam64_id}]

                soup = BeautifulSoup(response.text, 'html.parser')
                review_boxes = soup.find_all('div', class_='review_box')
                if not review_boxes:
                    if page == 1:
                        return [{"Status": "No reviews found", "UserID": steam64_id}]
                    return all_reviews

                for review in review_boxes:
                    try:
                        link_tag = review.find('div', class_='leftcol').find('a')
                        game_url = link_tag['href'] if link_tag else ""
                        game_name = "N/A"
                        genre_str = "N/A"

                        m = re.search(r'/app/(\d+)', game_url)
                        if m:
                            appid = m.group(1)
                            api_url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
                            try:
                                api_resp = requests.get(api_url, headers=headers, timeout=15)
                                if api_resp.status_code == 200:
                                    data = api_resp.json()
                                    if data.get(appid, {}).get('success'):
                                        game_data = data[appid]['data']
                                        game_name = game_data.get('name', 'N/A')
                                        genres = game_data.get('genres', [])
                                        genre_list = [g['description'] for g in genres]
                                        genre_str = ', '.join(genre_list) if genre_list else 'N/A'
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
                        print(f"Error parsing review: {str(e)}")
                        continue

                next_btn = soup.find('a', class_='pagebtn', string='>')
                if not next_btn:
                    return all_reviews

                success = True
                page += 1

            except requests.exceptions.RequestException as e:
                attempt += 1
                print(f"Attempt {attempt} failed for page {page}: {str(e)}")
                if attempt >= MAX_ATTEMPTS:
                    print(f"Max attempts reached for {steam64_id}")
                    return all_reviews if all_reviews else [{"Error": "Max attempts reached", "UserID": steam64_id}]
                time.sleep(delay * 2)

    print(f"Warning: Reached max page limit ({MAX_PAGES}) for {steam64_id}")
    return all_reviews

def process_csv(input_path):
    df = pd.read_csv(input_path)
    if 'Steam ID' not in df.columns:
        print(f"File {input_path} tidak memiliki kolom 'Steam ID'")
        return None

    all_reviews = []
    for steam_id in df['Steam ID'].dropna().unique():
        steam_id = str(steam_id).strip()
        if len(steam_id) != 17 or not steam_id.isdigit():
            print(f"ID tidak valid: {steam_id}")
            continue

        print(f"\nMemproses SteamID: {steam_id}")
        reviews = scrape_user_reviews(steam_id)
        if reviews:
            all_reviews.extend(reviews)

        time.sleep(get_random_delay())

    return pd.DataFrame(all_reviews) if all_reviews else None

def get_game_reviews(steam_ids: list):
    all_reviews = []
    for idx, steam_id in enumerate(steam_ids):
        print(f"\n🔄 [{idx+1}/{len(steam_ids)}] Memproses Steam ID: {steam_id}")
        # Pastikan steam_id adalah string
        if not isinstance(steam_id, str):
            print(f"⚠️ Steam ID {steam_id} tidak valid, harus berupa string.")
            continue
        raw_reviews = scrape_user_reviews(steam_id)
        all_reviews.extend(raw_reviews)
    if not all_reviews:
        return all_reviews
    return all_reviews

def main():
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    if not csv_files:
        print(f"Tidak ditemukan file CSV di folder {INPUT_DIR}")
        return

    for input_file in csv_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(OUTPUT_DIR, f"reviews_{filename}")

        print(f"\nMemproses file: {filename}")
        result_df = process_csv(input_file)

        if result_df is not None:
            result_df.to_csv(output_file, index=False)
            print(f"✅ Data disimpan di: {output_file}")
        else:
            print(f"❌ Tidak ada data yang berhasil di-scrape dari {filename}")

if __name__ == "__main__":
    main()
