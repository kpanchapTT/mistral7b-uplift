import os
import requests


def test_api_call():
    # read environment vars

    # export DEPLOY_URL="https://llm-chat--tenstorrent-playground.workload.tenstorrent.com"
    # export LLM_CHAT_API_URL="${DEPLOY_URL}/predictions/falcon40b/"

    # API_URL = os.environ["LLM_CHAT_API_URL"]
    API_URL = "http://127.0.0.1:1223/predictions/falcon40b"

    # headers = {"Authorization": os.environ.get('AUTHORIZATION')}
    headers = {}

    # set API prompt and optional parameters
    json_data = {
        "text": "Where should I go in Austin when I visit?",
        "temperature": 1,
        "top_k": 40,
        "top_p": 0.9,
        "max_tokens": 1,
        "stop_sequence": ".",
    }
    # using requests stream=True, make sure to set a timeout
    response = requests.post(
        API_URL,
        json=json_data,
        headers=headers,
        stream=True,
        timeout=30
    )
    # Handle chunked response
    if response.headers.get('transfer-encoding') == 'chunked':
        print("processing chunks ...")
        for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
            # Process each chunk of data as it's received
            print(chunk.decode(encoding="utf-8"))
    else:
        # If not chunked, you can access the entire response body at once
        print(response.text)


if __name__ == "__main__":
    test_api_call()