"""
Firebase Functions - Hybrid Architecture
Lightweight coordination layer for Cloud Run backend
"""

from firebase_functions import https_fn, storage_fn, options
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app, firestore, storage
import json
import requests

# Cost control
set_global_options(max_instances=10)

# Initialize Firebase Admin
initialize_app()

# Cloud Run backend URL - UPDATED TO CORRECT REGION
CLOUD_RUN_URL = "https://ekg-analysis-backend-140492000519.europe-west1.run.app"

@https_fn.on_request()
def hello_world(req: https_fn.Request) -> https_fn.Response:
    """Test function"""
    return https_fn.Response("LifeLine Hybrid EKG Analysis - Ready!")

@https_fn.on_request()
def analyze_ekg_manual(req: https_fn.Request) -> https_fn.Response:
    """Manual EKG analysis endpoint - forwards to Cloud Run"""
    try:
        # CORS headers
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Content-Type': 'application/json'
        }
        
        if req.method == 'OPTIONS':
            return https_fn.Response('', status=200, headers=headers)
        
        if req.method != 'POST':
            return https_fn.Response('Method not allowed', status=405, headers=headers)
        
        # Get request data
        request_json = req.get_json()
        if not request_json or 'file_path' not in request_json:
            return https_fn.Response(
                json.dumps({'error': 'file_path gerekli'}),
                status=400,
                headers=headers
            )
        
        # Forward to Cloud Run backend
        response = requests.post(
            f"{CLOUD_RUN_URL}/analyze",
            json=request_json,
            timeout=600  # 10 minutes timeout
        )
        
        # Return Cloud Run response
        return https_fn.Response(
            response.text,
            status=response.status_code,
            headers=headers
        )
        
    except requests.exceptions.Timeout:
        return https_fn.Response(
            json.dumps({'error': 'Analysis timeout - dosya çok büyük olabilir'}),
            status=408,
            headers=headers
        )
    except Exception as e:
        return https_fn.Response(
            json.dumps({'error': f'Proxy error: {str(e)}'}),
            status=500,
            headers=headers
        )

@storage_fn.on_object_finalized()
def process_ekg_file(cloud_event: storage_fn.CloudEvent[storage_fn.StorageObjectData]) -> None:
    """
    Lightweight storage trigger - forwards processing to Cloud Run
    """
    try:
        # Get file information
        file_data = cloud_event.data
        file_path = file_data.name
        
        print(f"File uploaded: {file_path}")
        
        # Check if this is an EKG file
        if not file_path.startswith('ekg/'):
            print("Not an EKG file, skipping")
            return
        
        # Forward to Cloud Run for heavy processing
        try:
            response = requests.post(
                f"{CLOUD_RUN_URL}/analyze",
                json={'file_path': file_path},
                timeout=600  # 10 minutes for large files
            )
            
            analysis_result = response.json()
            
        except requests.exceptions.Timeout:
            print(f"Analysis timeout for {file_path}")
            analysis_result = {
                'success': False,
                'error': 'Analysis timeout - dosya çok büyük'
            }
        except Exception as e:
            print(f"Cloud Run error: {e}")
            analysis_result = {
                'success': False,
                'error': f'Backend error: {str(e)}'
            }
        
        # Update Firestore with results
        db = firestore.client()
        file_name = file_path.split('/')[-1]
        
        # Find the corresponding document in analyses collection
        path_parts = file_path.split('/')
        if len(path_parts) >= 4:
            # Query for the analysis document
            analyses_ref = db.collection('analyses')
            query = analyses_ref.where('storagePath', '==', file_path).limit(1)
            docs = query.stream()
            
            for doc in docs:
                doc_ref = analyses_ref.document(doc.id)
                
                if analysis_result['success']:
                    doc_ref.update({
                        'analysisStatus': 'completed',
                        'analysisResult': analysis_result['analysis_result'],
                        'processed': True,
                        'processedAt': firestore.SERVER_TIMESTAMP,
                        'processedBy': 'cloud_run_backend'
                    })
                    print(f"Analysis completed successfully for {file_name}")
                else:
                    doc_ref.update({
                        'analysisStatus': 'failed',
                        'analysisError': analysis_result['error'],
                        'processed': False,
                        'processedAt': firestore.SERVER_TIMESTAMP,
                        'processedBy': 'cloud_run_backend'
                    })
                    print(f"Analysis failed for {file_name}: {analysis_result['error']}")
                break
            else:
                print(f"No matching analysis document found for {file_path}")
        
    except Exception as e:
        print(f"Storage trigger error: {e}")

@https_fn.on_request()
def get_backend_status(req: https_fn.Request) -> https_fn.Response:
    """Check Cloud Run backend status"""
    try:
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        }
        
        # Check Cloud Run health
        response = requests.get(f"{CLOUD_RUN_URL}/health", timeout=10)
        backend_status = response.json() if response.status_code == 200 else {'status': 'unhealthy'}
        
        return https_fn.Response(
            json.dumps({
                'functions_status': 'healthy',
                'backend_status': backend_status,
                'backend_url': CLOUD_RUN_URL,
                'architecture': 'hybrid'
            }),
            headers=headers
        )
        
    except Exception as e:
        return https_fn.Response(
            json.dumps({
                'functions_status': 'healthy',
                'backend_status': {'status': 'unreachable', 'error': str(e)},
                'backend_url': CLOUD_RUN_URL,
                'architecture': 'hybrid'
            }),
            headers=headers
        )