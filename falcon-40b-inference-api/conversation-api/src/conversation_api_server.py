import os
import uuid
from threading import Lock
import collections
import json
from datetime import timedelta

import requests
from flask import Flask, Response, jsonify, request, session

from conversation_memory import ConversationMemory
from conversation_config import conversation_config

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET", "test-secret-83492")
# set lifetime of the session cookie
app.permanent_session_lifetime = timedelta(minutes=15)
TT_API_KEY = os.getenv("TT_API_KEY", None)
TT_FALCON_40B_INFERENCE_API_URL = os.getenv("TT_FALCON_40B_INFERENCE_API_URL", None)


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
    output_buffer = [out for out in output_buffer if out != "<|endoftext|>"]
    with context.lock:
        context.conversations[session_id].save(
            {"input": prompt, "output": "".join(output_buffer)}
        )


def get_conversation_state(prompt, session_id):
    # get conversation in format
    with context.lock:
        state = context.conversations[session_id].load()
    if not state:
        state = ""
    return state


# def read_authorization(self) -> Optional[dict]:
#     [scheme, parameters] = normalize_token(self.headers.get("authorization", ""))
#     if scheme != "bearer":
#         self.log_error(f"Authorization scheme was '{scheme}' instead of bearer")
#         return None
#     try:
#         payload = jwt.decode(parameters, self.jwt_secret, algorithms=["HS256"])
#         return payload
#     except InvalidTokenError as exc:
#         self.log_error(f"JWT payload decode error: {exc}")
#         return None


@app.route("/conversation/falcon-40b-instruct", methods=["POST"])
def chat():
    json_data = json.loads(request.data.decode())
    prompt = json_data["text"]
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        conversation_state = f"Human: {prompt}\nAI: "
    else:
        conversation_state = get_conversation_state(
            prompt, session_id=session["session_id"]
        )
        conversation_state += f"Human: {prompt}\nAI: "

    # remove EOS token to avoid issues with deployed inference API
    conversation_state = conversation_state.replace("<|endoftext|>", "")
    print(f"sending: {conversation_state}")
    json_data["text"] = conversation_state
    session_id = session["session_id"]

    # need to replace Authorization header with subscription to APIM for inference API
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "Authorization": TT_API_KEY,
    }
    res = requests.post(
        TT_FALCON_40B_INFERENCE_API_URL,
        data=json.dumps(json_data),
        headers=headers,
        stream=True,
        timeout=600,
    )
    return Response(
        process_response(res, session_id, prompt=prompt),
        content_type="text/event-stream",
    )


def create_server():
    global app
    return app


if __name__ == "__main__":
    from flask_cors import CORS

    app = create_server()
    # CORS for swagger-ui local testing
    CORS(
        app,
        supports_credentials=True,
        resources={r"/conversation/*": {"origins": "http://localhost:8080"}},
    )
    app.run(debug=False, port=conversation_config.backend_server_port, host="0.0.0.0")
