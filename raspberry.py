from flask import Flask
import requests
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
host_url = os.getenv("HOST_URL")

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return "Raspberry Pi Online"

@app.route('/start-camera', methods=['GET'])
def start_camera():
    image = "hi there"
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.post(f'{host_url}/process-image', headers=headers, files={'image': image})
    return response.content

def capture_image():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
