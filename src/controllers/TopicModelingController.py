from flask import request, jsonify
from flask import Blueprint
from bertopic import BERTopic
from src.services.ScrappingService import ScrappingService
from src.utils.preprocessing.text_processor import preprocess_single_text
from src.utils.scraping.steam_data import get_reviews_from_steam_ids
from src.utils.scraping.steam_review import get_game_reviews
import io
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

AnalyzeApp = Blueprint('AnalyzeApp', __name__,)
scrappingService = ScrappingService()

def LoadModel():
    # Tentukan path model
    MODEL_PATH = 'public/models/player_topic_model'
    print(f"🔍 Mencoba memuat model dari: {MODEL_PATH}")
    topic_model = None
    try:
        topic_model = BERTopic.load(MODEL_PATH, embedding_model="all-MiniLM-L6-v2")
        print("✅ Model BERTopic berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model BERTopic: {e}")
    return topic_model

@AnalyzeApp.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Analyze App is running!"})

@AnalyzeApp.route('/scrapping', methods=['POST'])
def scrapping():
    data = request.json
    steam_ids = data.get('steam_ids', [])

    if not steam_ids or not isinstance(steam_ids, list):
        return jsonify({'error': 'Invalid input. Please provide a list of Steam IDs.'}), 400
    return jsonify({"message": "Analyze App is running!"})

# --- Endpoint untuk Topic Modeling dari Steam ID ---
@AnalyzeApp.route('/steam_id', methods=['POST'])
def analyze_steam_reviews():
    topic_model = LoadModel()
    if topic_model is None:
        return jsonify({'error': 'Model is not available.'}), 503

    data = request.json
    steam_ids = data.get('steam_ids', [])

    if not steam_ids or not isinstance(steam_ids, list):
        return jsonify({'error': 'Invalid input. Please provide a list of Steam IDs.'}), 400

    # 1. Scraping Review dari Steam API
    reviews_df  = get_game_reviews(steam_ids)
    if reviews_df.empty: 
        return jsonify({'message': 'No reviews found for the provided Steam IDs.'}), 404
    reviews_df.to_csv('raw_steam_reviews.csv', index=False)  # Simpan raw reviews
    list_of_reviews = reviews_df.to_dict('records')
    scrappingService.createNewScrapping(list_of_reviews)

    # reviews_df = pd.DataFrame(raw_reviews)
    reviews_df = reviews_df.dropna(subset=['Review'])
    reviews = reviews_df['Review'].tolist()

    reviews_df.to_csv('steam_reviews.csv', index=False) 
    print(f"✅ {len(reviews)} review berhasil diambil dari Steam API.")
    # 2. Preprocessing Review
    print(f"⚙️ Memproses {len(reviews)} review...")
    cleaned_reviews = [preprocess_single_text(review) for review in reviews]

    # 3. Topic Modeling
    topics, probabilities = topic_model.transform(cleaned_reviews)

    # 4. Susun Hasil Akhir (gabungan topik dan kata kunci)
    # Ini adalah bagian yang paling penting untuk menampilkan output yang bermakna
    unique_topics = sorted(list(set(topics)))
    results = []
    
    for topic_id in unique_topics:
        # Abaikan topic -1 (outliers)
        if topic_id == -1:
            continue
            
        topic_info = topic_model.get_topic(topic_id)
        topic_keywords = [word for word, score in topic_info]
        
        # Cari contoh review untuk topik ini
        sample_reviews_for_topic = [
            reviews[i] for i, t in enumerate(topics)
            if t == topic_id
        ][:3] # Ambil 3 contoh saja
        
        results.append({
            'topic_id': int(topic_id),
            'keywords': topic_keywords,
            'number_of_reviews': topics.tolist().count(topic_id),
            'sample_reviews': sample_reviews_for_topic
        })

    return jsonify({'topics': results})


@AnalyzeApp.route('/steam_reviews', methods=['POST'])
def upload_and_analyze():
    topic_model = LoadModel()
    # 1. Cek apakah model sudah dimuat
    if topic_model is None:
        return jsonify({'error': 'Model is not available. Please check the server logs.'}), 503

    # 2. Cek apakah ada file yang diunggah
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # 3. Proses file jika formatnya CSV
    if file and file.filename.endswith('.csv'):
        try:
            # Baca file CSV langsung ke pandas DataFrame
            file_stream = io.StringIO(file.stream.read().decode("utf-8"))
            df = pd.read_csv(file_stream)
            df.to_csv('uploaded_reviews.csv', index=False)  # Simpan file yang diunggah
            # Validasi kolom 'Review'
            if 'Review' not in df.columns:
                return jsonify({'error': "File must contain a 'Review' column."}), 400
                
            # Handle NaN values
            reviews_df = df.dropna(subset=['Review'])
            raw_reviews = reviews_df['Review'].tolist()

            if not raw_reviews:
                return jsonify({'message': 'No valid reviews found in the file.'}), 404

            # 4. Preprocessing dan Topic Modeling
            print(f"⚙️ Memproses {len(raw_reviews)} review dari file yang diunggah...")
            cleaned_reviews = [preprocess_single_text(review) for review in raw_reviews]
            pd.Series(cleaned_reviews).to_csv('cleaned_reviews.csv', index=False)  # Simpan cleaned reviews
            topics, probabilities = topic_model.transform(cleaned_reviews)

            # 5. Susun hasil akhir
            results = []
            for topic_id in sorted(list(set(topics))):
                if topic_id == -1: continue # Abaikan outlier

                topic_info = topic_model.get_topic(topic_id)
                topic_keywords = [word for word, score in topic_info]

                # Ambil contoh review dari file yang diunggah
                sample_reviews_for_topic = reviews_df[topics == topic_id].head(3).to_dict('records')

                results.append({
                    'topic_id': int(topic_id),
                    'keywords': topic_keywords,
                    'number_of_reviews': topics.tolist().count(topic_id),
                    'sample_reviews_data': [
                        {'Game': r.get('Game', 'N/A'), 'Review': r.get('Review', 'N/A')}
                        for r in sample_reviews_for_topic
                    ]
                })

            return jsonify({'topics': results})

        except Exception as e:
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400



def mmr(doc_embedding, word_embeddings, words, top_n=10, diversity=0.7):
    LoadModel = LoadModel()
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding.reshape(1, -1))
    word_similarity = cosine_similarity(word_embeddings)
    
    selected_words = []
    selected_idxs = []
    remaining_idxs = list(range(len(words)))

    selected_idxs.append(np.argmax(word_doc_similarity))
    selected_words.append(words[selected_idxs[0]])
    remaining_idxs.remove(selected_idxs[0])


@AnalyzeApp.route('/analyze', methods=['POST'])
def analyze():
    # 1. Cek apakah model sudah dimuat
    if topic_model is None:
        return jsonify({'error': 'Model is not available. Please check the server logs.'}), 503

    # 2. Cek apakah ada file yang diunggah
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # 3. Proses file jika formatnya CSV
    if file and file.filename.endswith('.csv'):
        try:
            # Baca file CSV langsung ke pandas DataFrame
            file_stream = io.StringIO(file.stream.read().decode("utf-8"))
            df = pd.read_csv(file_stream)
            df.to_csv('uploaded_reviews.csv', index=False)  # Simpan file yang diunggah
            # Validasi kolom 'Review'
            if 'Review' not in df.columns:
                return jsonify({'error': "File must contain a 'Review' column."}), 400
                
            # Handle NaN values
            reviews_df = df.dropna(subset=['Review'])
            raw_reviews = reviews_df['Review'].tolist()

            if not raw_reviews:
                return jsonify({'message': 'No valid reviews found in the file.'}), 404

            # 4. Preprocessing dan Topic Modeling
            print(f"⚙️ Memproses {len(raw_reviews)} review dari file yang diunggah...")
            cleaned_reviews = [preprocess_single_text(review) for review in raw_reviews]
            pd.Series(cleaned_reviews).to_csv('cleaned_reviews.csv', index=False)  # Simpan cleaned reviews
            topics, probabilities = topic_model.transform(cleaned_reviews)
            # 2. Siapkan embedding
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedding_model.encode(cleaned_reviews, show_progress_bar=True)

            # 6. Ekstrak topik dan hitung MMR
            topics_keywords = {}
            for topic in topic_model.get_topics().keys():
                if topic == -1:
                    continue
                topic_words_scores = topic_model.get_topic(topic)
                words = [w for w, _ in topic_words_scores]
                embeddings_words = embedding_model.encode(words, show_progress_bar=False)
                topic_embedding = np.mean(embedding_model.encode([" ".join(words)]), axis=0)
                top_keywords = mmr(topic_embedding, embeddings_words, words, top_n=10, diversity=0.7)
                topics_keywords[topic] = top_keywords

            # 7. Simpan topik ke CSV
            topic_keywords_df = pd.DataFrame([
                {"Cluster": k, "Top_Keywords": v}
                for k, v in topics_keywords.items()
            ])
            topic_keywords_df.to_csv("top_topic_keywords.csv", index=False)

            return jsonify({'topics': topic_keywords_df.to_dict(orient='records')})

        except Exception as e:
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

