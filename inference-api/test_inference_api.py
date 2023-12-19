import os
import threading

import requests
from inference_config import inference_config

DEPLOY_URL = "http://127.0.0.1"
# DEPLOY_URL="https://llm-chat--tenstorrent-playground.workload.tenstorrent.com"
API_BASE_URL = f"{DEPLOY_URL}:{inference_config.reverse_proxy_port}"
API_URL = f"{API_BASE_URL}/predictions/falcon40b"
HEALTH_URL = f"{API_BASE_URL}/get-health"

headers = {"Authorization": os.environ.get("AUTHORIZATION")}


def test_api_call(prompt_extra="", print_output=True):
    # set API prompt and optional parameters
    json_data = {
        "text": "Where should I go in Austin when I visit?" + prompt_extra,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.9,
        "max_tokens": 16,
        "stop_sequence": None,
        "return_prompt": None,
    }
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL, json=json_data, headers=headers, stream=True, timeout=35
    )
    # Handle chunked response
    if response.headers.get("transfer-encoding") == "chunked":
        print("processing chunks ...")
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            # Process each chunk of data as it's received
            if print_output:
                print(chunk)
    else:
        # If not chunked, you can access the entire response body at once
        print(response.text)


def test_api_call_threaded():
    threads = []

    for i in range(128):
        thread = threading.Thread(target=test_api_call, args=[str(i), False])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished execution.")


def test_get_health():
    response = requests.get(HEALTH_URL, headers=headers, timeout=35)
    assert reponse


if __name__ == "__main__":
    # test_get_health()
    test_api_call()
    # test_api_call_threaded()
