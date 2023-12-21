import time
import requests
from dotenv import load_dotenv
import os

load_dotenv()

host_url = os.getenv("HOST_URL")

def poll_raspberry_pi():
    try:
        response = requests.get(f'{host_url}/check-edge')
        return response.status_code == 200
    except (requests.ConnectionError, requests.ConnectionRefusedError):
        return False

def start_oak_camera():
    response = requests.get(f'{host_url}/start-oak-camera')
    return response.content

if __name__ == "__main__":
    while True:
        is_available = poll_raspberry_pi()
        if is_available:
            print("Raspberry Pi Online. Starting Camera")
            camera_response = start_oak_camera()
            print("Camera Response: ", camera_response)
            break
        else:
            print("Raspberry Pi not Online. Retrying in 10 seconds")
            time.sleep(10)
