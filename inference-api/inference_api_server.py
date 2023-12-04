from flask import Flask, request, jsonify, session, Response
# from flask_socketio import SocketIO, emit, join_room, leave_room
import multiprocessing
import threading
import uuid
from threading import Lock

import time
import sys
import os
sys.path.append(os.getcwd())

# from decode_backend_v0 import run_decode_backend
from _mock_decode_backend import run_decode_backend
from inference_config import inference_config

app = Flask(__name__)
app.secret_key = 'your_secret_key'
INIT_ID = 'COMPILE-INITIALIZATION'
# socketio = SocketIO(app)

# Store current context
# Store conversation history
# Initialize the lock

class Context:
    def __init__(self):
        self.conversations = {}
        self.user_status={} # {user_id:q_position}
        self.num_decoding_users=0
        self.user_parameters={}
        # Initialize the lock
        self.conversations_lock = Lock()

context = Context()

# 1L pytorch no weights
# override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
#                  'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
#                  '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9',
#                  ]

# # # 60L pytorch
# # 1L pytorch no weights
# override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
#                  'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
#                  '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', #'--top-k', '5', '--top-p', '0.9',
#                  ]

# # 60L pytorch
# override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
#                  'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
#                  '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load-pretrained',
#                  '--model', 'tiiuae/falcon-40b-instruct'
#                  ]

# # 2L silicon
override_args = ['--mode', 'concurrent', '-l', '2', '--version', 'efficient-40b', '-d',
                 'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--num-outer-loops', '100_000',
                 '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
                 '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
                 '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load', 'flash_decode_2l_v0_test.tti',
                 '--load-pretrained', '--model', 'tiiuae/falcon-40b-instruct',
                 ]

# # 60L silicon
# override_args = ['--mode', 'concurrent', '-l', '60', '--version', 'efficient-40b', '-d',
#                  'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1000000', '--num-outer-loops', '1000',
#                  '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load', 'flash_decode_60l_v0_instruct.tti',
#                  '--load-pretrained', '--model', 'tiiuae/falcon-40b-instruct',
#                  ]

# # 60L silicon instruct
# override_args = ['--mode', 'concurrent', '-l', '60', '--version', 'efficient-40b', '-d',
#                  'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1000000', '--num-outer-loops', '1000',
#                  '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load', 'flash_decode_60l_v0_instruct.tti',
#                  '--load-pretrained', '--model', 'tiiuae/falcon-40b-instruct',
#                  ]

verbose = False

def initialize_decode_backend():
    global input_queue
    global output_queue
    global status_queue
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    # run the decode backend in a separate process
    p = multiprocessing.Process(target=run_decode_backend, args=(input_queue, output_queue, status_queue, override_args, verbose))
    p.start()
    input_queue.put((INIT_ID, 'Dummy input for initialization', get_user_parameters({})))
    # respond_to_users_thread = threading.Thread(target=respond_to_users)
    # respond_to_users_thread.start()
    # poll_status_thread = threading.Thread(target=poll_status)
    # poll_status_thread.start()

def validate_request(request):
    error = None
    if request.is_json:
        data = request.get_json()
    else:
        error = 'Request was not JSON', 400
        return None, error

    if not data.get("text"):
        error = "required 'text' parameter is either empty or not given", 400
    return data, error

def get_user_parameters(data):
    default_temperature = 1.0
    default_top_p = 0.9
    default_top_k = 20
    default_max_tokens = 128
    default_stop_sequence = None  # EOS
    params = {
        "temperature": data.get("temperature", default_temperature),
        "top_p": data.get("top_p", default_top_p),
        "top_k": data.get("top_k", default_top_k),
        "max_tokens": data.get("max_tokens", default_max_tokens),
        "stop_sequence": data.get("stop_sequence", default_stop_sequence),
    }
    return params


def get_output(session_id):
    done_generation = False
    while not done_generation:
        if output_queue.empty():
            time.sleep(0.1)
            continue
        
        response_session_id, out_text = output_queue.get_nowait()

        # Log response
        with open(f'server_logs/{response_session_id}.txt', 'a') as f:
            f.write(out_text)
        if response_session_id == INIT_ID:
            continue

        if response_session_id == session_id:  # Check for specific key
            if out_text == "<|endoftext|>":
                done_generation = True
            yield out_text


@app.route('/predictions/falcon40b', methods=['POST'])
def inference():
    data, error = validate_request(request)
    if error:
        return error

    # create a session_id if not supplied
    if "session_id" not in data:
        session.clear()
        session['session_id'] = str(uuid.uuid4())
        # with context.conversations_lock:
        #     conversation = context.conversations.get(session['session_id'], '')
    
    # if full, wait n seconds, exponential back-off
    #   if still full after N seconds, respond with busy signal
    # elif has capacity
    # input
    session_id = session.get('session_id')
    user_message = data["text"]
    user_message = _preprocess_prompt(user_message)
    params = get_user_parameters(data)
    input_queue.put((session_id, user_message, params))
    # with context.conversations_lock:
    #     context.conversations[session_id] = ''
    #     context.user_parameters = get_user_parameters(data)

    # Log user's prompt
    with open(f'server_logs/{session_id}.txt', 'a') as f:
        f.write('Prompt:\n' + user_message + '\n')

    # output
    return Response(get_output(session_id), content_type='text/event-stream')

def _preprocess_prompt(prompt):
    preprocessed_prompt = f"User: {prompt}\nAI:"
    return preprocessed_prompt

if __name__ == '__main__':
    # Create server log directory
    if not os.path.exists('server_logs'):
        os.makedirs('server_logs')
    initialize_decode_backend()
    app.run(debug=True, port=1223, host='0.0.0.0')
