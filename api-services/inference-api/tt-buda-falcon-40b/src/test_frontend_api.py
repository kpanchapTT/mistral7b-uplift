import json
import os

import requests
from flask import Flask, Response, request
from inference_config import inference_config

API_URL = (
    f"http://127.0.0.1:{inference_config.backend_server_port}/predictions/falcon40b"
)
HEADERS = {"Authorization": os.environ.get("AUTHORIZATION")}


def front_end_call():
    def call_server():
        def process_response():
            json_data = {"text": "Where is the best coffee? ", "max_tokens": 10}
            response = requests.post(
                API_URL, json=json_data, headers=HEADERS, stream=True, timeout=30
            )
            print(response)
            full_message = ""
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                print(chunk)
                # spaces must be added after each chunk
                if chunk != "<|endoftext|>":
                    full_message += chunk
                    yield f"event: answer\ndata: {json.dumps({'message': full_message})}\n\n"

            yield 'event: close\ndata: {"message": "Connection closed"}\n\n'

        return Response(process_response(), content_type="text/event-stream")

    res = call_server()
    for chunk in res.iter_encoded():
        print(chunk)


if __name__ == "__main__":
    front_end_call()
