import os
import uuid
from threading import Lock
import collections

import requests
from flask import Flask, Response, jsonify, request, session

from conversation_memory import ConversationMemory
from conversation_config import conversation_config

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "test-secret-83492")
TT_API_KEY = os.getenv("TT_API_KEY", None)
TT_API_URL = os.getenv("TT_API_URL", None)


class Context:
    def __init__(self):
        # {session_id : ConversationMemory}
        self.conversations = collections.defaultdict(ConversationMemory)
        self.lock = Lock()
    def load(self, session_id):
        self.conversations[session_id].load()

    def save(self, session_id, text):
        self.conversations[session_id].save(text)

    def clear(self, session_id):
        self.conversations[session_id].clear()

# Shared variables with a lock for thread-safe access
context = Context()

def process_response(response, session_id, prompt):
    output_buffer = []
    # stream output
    for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
        decoded_chunk = chunk.decode(encoding="utf-8")
        output_buffer.append(decoded_chunk)
        yield decoded_chunk
    # after recieved all chunks save conversation history
    with context.lock:
        context.conversations[session_id].save({
            "input": prompt,
            "output": "".join(output_buffer)
        })
    

@app.route("/conversation", methods=["POST, OPTIONS"])
def chat():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    else:
        print(f"PRE-EXISTING SESSION: {session.get('session_id')}")
        # TODO: add user_session_id as session_id if passed correctly
        # currently only support stateless sessions
        # session["session_id"] = str(uuid.uuid4())
        breakpoint()
    session_id = session["session_id"]
    json_data = request.data
    # need to replace Authorization header with subscription to APIM for inference API
    headers = request.headers
    headers["authorization"] = ""

    res = requests.post(TT_API_URL, json=json_data, headers=request.headers, stream=True, timeout=600)
    return Response(process_response(res, session_id), content_type='text/event-stream')

def create_server():
    return app

if __name__ == '__main__':
    app = create_server()
    app.run(
        debug=True,
        port=conversation_config.backend_server_port,
        host="0.0.0.0"
    )
