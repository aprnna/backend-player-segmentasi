from flask import request, jsonify, g
from flask import Blueprint
from bertopic import BERTopic
from src.services.ScrappingService import ScrappingService
from src.services.TopicModelingService import TopicModelingService
from src.utils.preprocessing.text_processor import preprocess_single_text
from src.utils.scraping.steam_data import get_reviews_from_steam_ids
from src.utils.scraping.steam_review import get_game_reviews
import src.utils.getResponse as Response
import io
import os
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from src.middlewares.AuthMiddleware import isAuthenticated

AnalyzeApp = Blueprint('AnalyzeApp', __name__,)
scrappingService = ScrappingService()
topicModelingService = TopicModelingService()

@AnalyzeApp.route('/test', methods=['GET'])
def test():
    return Response.success("Analyze App is running!", "success")

@AnalyzeApp.route('/scrapping', methods=['POST'])
def scrapping():
    data = request.json
    steam_ids = data.get('steam_ids', [])

    if not steam_ids or not isinstance(steam_ids, list):
        return Response.error("Invalid input. Please provide a list of Steam IDs.", 400)
    return Response.success(scrappingService.createNewScrapping(steam_ids), "success scrapping data")

# --- Endpoint untuk Topic Modeling dari Steam ID ---
@AnalyzeApp.route('/steam_id', methods=['POST'])
@isAuthenticated
def analyze_steam_reviews():
    data = request.json
    steam_ids = data.get('steam_ids', [])
    if not steam_ids or not isinstance(steam_ids, list):
        return Response.error({'error': 'Invalid input. Please provide a list of Steam IDs.'}), 400
    result = topicModelingService.createNewTopicModeling(steam_ids, g.user['user_id'])
    return Response.success(result, "success analyze steam reviews")

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


