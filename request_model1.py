import requests
import time

url = "http://localhost:8081/predictions/cross_encoder_model1"
headers = {"Content-Type": "application/json"}
data = '{"body": "What is the capital of France?"}'

try:
    while True:
        response = requests.post(url, headers=headers, data=data)
        print(f"Status Code: {response.status_code}, Response: {response.text}")
        time.sleep(1)  # Wait for 1 second before sending the next request
except KeyboardInterrupt:
    print("Stopped sending requests.")