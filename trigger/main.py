import os
import logging
from google.cloud import run_v2
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def trigger_job():
    try:
        logger.info("Starting job trigger")
        
        # Create the client
        client = run_v2.JobsClient()
        
        # Get the full job name
        job_path = client.job_path(
            'senate-stock-rag-chatbot',
            'us-west1',
            'senate-stock-rag-update'
        )
        
        # Create and execute the job request
        request = run_v2.RunJobRequest(name=job_path)
        operation = client.run_job(request=request)
        
        # Don't wait for completion, just return success
        return jsonify({
            "status": "success",
            "message": f"Job triggered: {operation.operation.name}"
        }), 200
        
    except Exception as e:
        logger.exception("Error triggering job")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)