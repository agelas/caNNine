from flask import Flask, request, abort
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
raspberry_pi_url = os.getenv("RASPBERRY_PI_URL")
api_key = os.getenv("API_KEY")

@app.route('/start-oak-camera', methods=['GET'])
def start_oak_camera():
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(f'{raspberry_pi_url}/start-camera', headers=headers)
    return response.content

@app.route('/check-edge', methods=['GET'])
def check_edge():
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        response = requests.get(f'{raspberry_pi_url}/ping', headers=headers)
        return response.content
    except requests.ConnectionError:
        return "Rasperry Pi Not Online"

@app.route('/process-image', methods=['POST'])
def process_image():
    auth_header = request.headers.get('Authorization')
    if auth_header != f'Bearer {api_key}':
        abort(401)

    image = request.files['image']
    return "Processing not Implemented Yet"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
