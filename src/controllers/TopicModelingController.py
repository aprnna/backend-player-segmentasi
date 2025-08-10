from flask import request, jsonify, g
from flask import Blueprint
from src.services.ScrappingService import ScrappingService
from src.services.TopicModelingService import TopicModelingService
from src.services.SegmentasiService import SegmentationService
from src.services.AnalysisOrchestratorService import AnalysisOrchestratorService
from src.utils.preprocessing.text_processor import preprocess_single_text
from src.utils.scraping.steam_data import get_steam_id_data
from src.utils.scraping.steam_review import get_game_reviews
import src.utils.getResponse as Response
import io
import pandas as pd
import numpy as np
from src.middlewares.AuthMiddleware import isAuthenticated

AnalyzeApp = Blueprint('AnalyzeApp', __name__,)
scrappingService = ScrappingService()
topicModelingService = TopicModelingService()
segmentationService = SegmentationService()
analysisOrchestratorService = AnalysisOrchestratorService()

# --- Endpoint untuk Analisis Lengkap ---
@AnalyzeApp.route('/full_steam_id', methods=['POST'])
@isAuthenticated
def analyze_steam_data_full_pipeline():
    data = request.json
    steam_ids = data.get('steam_ids', [])
    if not steam_ids or not isinstance(steam_ids, list):
        return Response.error({'error': 'Input tidak valid. Harap berikan list dari Steam ID.'}), 400
    
    # Panggil satu fungsi dari service orkestrator
    result = analysisOrchestratorService.run_full_analysis_pipeline(steam_ids, g.user['user_id'])
    
    if result.get('status') == 'success':
        return Response.success(result.get('data'), "Analisis lengkap berhasil dijalankan.")
    else:
        return Response.error(result.get('data'), result.get('code', 500))


@AnalyzeApp.route('/test', methods=['GET'])
def test():
    return Response.success("Analyze App is running!", "success")

@AnalyzeApp.route('/segmentasi', methods=['POST'])
def segmentasi():
    data = request.json
    steam_ids = data.get('steam_ids', [])
    if not steam_ids or not isinstance(steam_ids, list):
        return Response.error("Invalid input. Please provide a list of Steam IDs.", 400)
    segmentationService.run_segmentation_pipeline(steam_ids)
    return Response.success("Segmentasi pipeline is running!", "success")

@AnalyzeApp.route('/scrapping-review', methods=['POST'])
def scrapping_review():
    data = request.json
    steam_ids = data.get('steam_ids', [])
    if not steam_ids or not isinstance(steam_ids, list):
        return Response.error("Invalid input. Please provide a list of Steam IDs.", 400)
    reviews = get_game_reviews(steam_ids)
    if not reviews:
        return Response.error("No reviews found for the provided Steam IDs.", 404)
    df = pd.DataFrame(reviews)
    df.to_csv('scraped_steam_reviews.csv', index=False)
    return Response.success(reviews, "success scrapping data")

@AnalyzeApp.route('/scrapping-steam-data', methods=['POST'])
def scrapping_steam_data():
    data = request.json
    steam_ids = data.get('steam_ids', [])
    steam_data = get_steam_id_data(steam_ids)
    # save csv
    if steam_data:
        df = pd.DataFrame(steam_data)
        df.to_csv('scraped_steam_data.csv', index=False)
    if not steam_data:
        return Response.error("No data found for the provided Steam IDs.", 404)
    return Response.success(steam_data, "success scrapping data")


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


