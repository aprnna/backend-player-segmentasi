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
import uuid
import threading
import time
from datetime import datetime, timedelta
from src.utils.uploadFIle import upload_file, delete_file
AnalyzeApp = Blueprint('AnalyzeApp', __name__,)
scrappingService = ScrappingService()
topicModelingService = TopicModelingService()
segmentationService = SegmentationService()
analysisOrchestratorService = AnalysisOrchestratorService()

from flask import current_app
import threading
import uuid
from datetime import datetime

# In-memory job storage
job_status = {}

@AnalyzeApp.route('/full_steam', methods=['POST'])
@isAuthenticated
def analyze_with_threading():
    try:
        steam_ids = []
        if request.form.get('steam_ids'):
            steam_ids = request.form['steam_ids'].split(", ")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_status[job_id] = {
            'status': 'started',
            'progress': 0,
            'message': 'Analysis started',
            'created_at': datetime.now(),
            'result': None,
            'error': None
        }
        
        # Get Flask app context
        app = current_app._get_current_object()  # 🔑 Key line
        files_dict = {}
        
        # Handle file upload properly
        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            csv_path = upload_file(uploaded_file)
            if not csv_path:
                return jsonify({"status": "failed", "message": "Gagal upload file"}), 400
            files_dict["file"] = csv_path  # Store only the file path
        else:
            files_dict["file"] = None  
        # Start background thread with app context
        print(f"🔄 Starting analysis for job: {files_dict}")
        thread = threading.Thread(
            target=run_analysis_with_context,
            args=(app, job_id, steam_ids, files_dict, g.user['user_id'])
        )
        thread.daemon = True  # Dies when main thread dies
        thread.start()
        
        return Response.success({
            'job_id': job_id,
            'status': 'processing',
            'message': 'Analysis started in background',
            'check_url': f'/analyze/status/{job_id}'
        }, "Threading analysis started successfully")
        
    except Exception as e:
        return Response.error(f"Failed to start analysis: {str(e)}", 500)

def run_analysis_with_context(app, job_id, steam_ids, files_dict, user_id):
    """Background function dengan proper Flask app context"""
    with app.app_context():  # 🔑 Critical: Setup app context
        try:
            print(f"🔄 Starting threaded analysis for job: {job_id}")
            
            # Update progress
            job_status[job_id].update({
                'status': 'processing',
                'progress': 10,
                'message': 'Initializing analysis...'
            })
            
            # Initialize service (now has app context)
            from src.services.AnalysisOrchestratorService import AnalysisOrchestratorService
            analysisOrchestratorService = AnalysisOrchestratorService()
            
 

            # Run analysis (this now works because of app context)
            result = analysisOrchestratorService.run_full_analysis_pipeline(
                steam_ids, files_dict, user_id  # Files handling might need adjustment
            )
            
            if result.get('status') == 'success':
                job_status[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Analysis completed successfully',
                    'result': result.get('data')
                })
            else:
                job_status[job_id].update({
                    'status': 'failed',
                    'error': result.get('data'),
                    'message': f'Analysis failed: {result.get("message")}'
                })
                
        except Exception as e:
            job_status[job_id].update({
                'status': 'failed',
                'error': str(e),
                'message': f'Analysis failed: {str(e)}'
            })
            print(f"❌ Threading analysis error: {e}")

@AnalyzeApp.route('/status/<job_id>', methods=['GET'])
@isAuthenticated
def check_thread_status(job_id):
    """Check threading job status"""
    if job_id not in job_status:
        return Response.error("Job not found", 404)
    
    return Response.success(job_status[job_id], "Job status retrieved")


@AnalyzeApp.route('/test', methods=['GET'])
@isAuthenticated
def test():
    return Response.success(g.user, "success")

@AnalyzeApp.route('/<proses_id>', methods=['GET'])
@isAuthenticated
def getAnalayzeProses(proses_id):
    data = analysisOrchestratorService.getDetailProses(proses_id, g.user['user_id'])
    if not data:
        return Response.error("No analysis process found.", 404)
    return Response.success(data['data'], "success")

@AnalyzeApp.route('', methods=['GET'])
@isAuthenticated
def getAnalyzeProses():
    data = analysisOrchestratorService.getAllProsesByUserId(g.user['user_id'])
    if not data:
        return Response.error("No analysis processes found.", 404)
    return Response.success(data['data'], "success")

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
