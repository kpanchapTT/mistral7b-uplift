import os
import uuid
from threading import Lock
import collections
import json
from datetime import timedelta
from typing import Optional

import requests
import jwt
from flask import Flask, Response, request, session, abort


from conversation_memory import ConversationMemory
from conversation_config import conversation_config

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET", "test-secret-83492")
# set lifetime of the session cookie
app.permanent_session_lifetime = timedelta(minutes=15)

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503


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


def normalize_token(token) -> [str, str]:
    """
    Note that scheme is case insensitive for the authorization header.
    See: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization#directives
    """  # noqa: E501
    one_space = " "
    words = token.split(one_space)
    scheme = words[0].lower()
    return [scheme, " ".join(words[1:])]


def read_authorization(
    headers,
) -> Optional[dict]:
    authorization = headers.get("authorization")
    if not authorization:
        abort(HTTP_UNAUTHORIZED, description="Must provide Authorization header.")
    [scheme, parameters] = normalize_token(authorization)
    if scheme != "bearer":
        user_error_msg = f"Authorization scheme was '{scheme}' instead of bearer"
        abort(HTTP_UNAUTHORIZED, description=user_error_msg)
    try:
        payload = jwt.decode(parameters, os.getenv("JWT_SECRET"), algorithms=["HS256"])
        if not payload:
            abort(HTTP_UNAUTHORIZED)
        return payload
    except jwt.InvalidTokenError as exc:
        user_error_msg = f"JWT payload decode error: {exc}"
        abort(HTTP_BAD_REQUEST, description=user_error_msg)


def handle_internal_api_errors(res):
    # send internal inference API errors to user
    if res.status_code != HTTP_OK:
        if res.status_code == HTTP_UNAUTHORIZED:
            # if the internal inference API authorization is broken, backend fix is required
            abort(
                HTTP_SERVICE_UNAVAILABLE,
                "Internal service issue is under investigation ...",
            )
        else:
            # all other errors can be passed through to user
            abort(res.status_code, description=res.content)


@app.route("/conversation/falcon-40b-instruct", methods=["POST"])
def chat():
    _ = read_authorization(request.headers)
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
    json_data["text"] = conversation_state
    session_id = session["session_id"]

    # need to replace Authorization header with subscription to APIM for inference API
    headers = {
        "accept": "*/*",
        "Content-Type": "application/json",
        "Authorization": os.getenv("TT_API_JWT", None),
    }
    res = requests.post(
        conversation_config.tt_falcon_40b_inference_api_url,
        data=json.dumps(json_data),
        headers=headers,
        stream=True,
        timeout=600,
    )
    handle_internal_api_errors(res)
    return Response(
        process_response(res, session_id, prompt=prompt),
        content_type="text/event-stream",
    )


def create_server():
    print("Starting converstaion API server ...")
    return app


def create_test_server():
    from flask_cors import CORS

    app = create_server()
    # CORS for swagger-ui local testing
    CORS(
        app,
        supports_credentials=True,
        resources={r"/conversation/*": {"origins": "http://localhost:8080"}},
    )
    return app


if __name__ == "__main__":
    app = create_test_server()
    app.run(debug=False, port=conversation_config.backend_server_port, host="0.0.0.0")
