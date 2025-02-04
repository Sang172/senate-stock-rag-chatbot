import os
from google.cloud import run_v2
import functions_framework
from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['POST'])
def trigger_job():
    # Hardcoded project details
    project_id = 'senate-stock-rag-chatbot'
    job_name = 'senate-stock-rag-update'
    region = 'us-west1'
    
    # Create the client
    client = run_v2.JobsClient()
    
    # Get the full job name
    job_path = client.job_path(project_id, region, job_name)
    
    # Create the run job request
    request = run_v2.RunJobRequest(
        name=job_path
    )
    
    # Execute the job
    operation = client.run_job(request=request)
    
    # Wait for the operation to complete
    result = operation.result()
    
    return f"Job triggered successfully: {result.name}", 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)