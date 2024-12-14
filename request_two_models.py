import requests
import time

url1 = "http://localhost:8081/predictions/cross_encoder_model1"
url2 = "http://localhost:8081/predictions/cross_encoder_model2"
headers = {"Content-Type": "application/json"}
data = '{"body": "What is the capital of France?"}'

try:
    while True:
        response1 = requests.post(url1, headers=headers, data=data)
        response2 = requests.post(url2, headers=headers, data=data)
        print(f"Status Code: {response1.status_code}, Response 1: {response1.text}")
        print(f"Status Code: {response2.status_code}, Response 2: {response2.text}")
        time.sleep(1)  # Wait for 1 second before sending the next request
except KeyboardInterrupt:
    print("Stopped sending requests.")