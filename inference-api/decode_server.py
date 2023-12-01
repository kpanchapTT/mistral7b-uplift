from flask import Flask, request, render_template, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import multiprocessing
import threading
import uuid
from threading import Lock

import sys
import os
sys.path.append(os.getcwd())

from decode_backend_v0 import run_decode_backend
from inference_config import inference_config

app = Flask(__name__)
app.secret_key = 'your_secret_key'
INIT_ID = 'COMPILE-INITIALIZATION'
socketio = SocketIO(app)

# Store current context
# Store conversation history
# Initialize the lock

class Context:
    def __init__(self):
        self.conversations = {}
        self.user_status={} # {user_id:q_position}
        self.num_decoding_users=0
        # Initialize the lock
        self.conversations_lock = Lock()

context = Context()

# 1L pytorch no weights
override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
                 'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
                 '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
                 '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', inference_config.hf_cache,
                 '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9',
                 ]

# # # 60L pytorch
# # 1L pytorch no weights
# override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
#                  'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
#                  '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', HF_CACHE,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', #'--top-k', '5', '--top-p', '0.9',
#                  ]

# # 60L pytorch
# override_args = ['--mode', 'concurrent', '-l', '1', '--version', 'efficient-40b', '-d',
#                  'pytorch', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--user-rows',
#                  '32', '--precision', 'fp32', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', HF_CACHE,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load-pretrained',
#                  '--model', 'tiiuae/falcon-40b-instruct'
#                  ]

# # 2L silicon
# override_args = ['--mode', 'concurrent', '-l', '2', '--version', 'efficient-40b', '-d',
#                  'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1_000_000_000', '--num-outer-loops', '100_000',
#                  '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', HF_CACHE,
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', #'--load', 'flash_decode_2l_v0.tti',
#                  #'--load-pretrained'
#                  ]

# # 60L silicon
# override_args = ['--mode', 'concurrent', '-l', '60', '--version', 'efficient-40b', '-d',
#                  'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1000000', '--num-outer-loops', '1000',
#                  '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', '/localdev/xuncai',
#                  '-odlmh', '-plmh', '-fv', '--flash-decode', '--top-k', '5', '--top-p', '0.9', '--load', 'flash_decode_60l_v0_instruct.tti',
#                  '--load-pretrained', --model tiiuae/falcon-40b-instruct,
#                  ]

# # 60L silicon instruct
# override_args = ['--mode', 'concurrent', '-l', '60', '--version', 'efficient-40b', '-d',
#                  'silicon', '--arch', 'nebula-galaxy', '--num-tokens', '1000000', '--num-outer-loops', '1000',
#                  '--user-rows', '32', '--precision', 'bf16', '--num-chips', '32', '-mf', '8',
#                  '--log-level', 'ERROR', '--opt-level', '4', '--hf-cache', '/localdev/xuncai',
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
    input_queue.put((INIT_ID, 'Dummy input for initialization'))
    respond_to_users_thread = threading.Thread(target=respond_to_users)
    respond_to_users_thread.start()
    poll_status_thread = threading.Thread(target=poll_status)
    poll_status_thread.start()



def _preprocess_prompt(prompt):
    preprocessed_prompt = f"User: {prompt}\nAI:"
    return preprocessed_prompt

@app.route('/', methods=['GET'])
def index():
    session.clear()  # Clear existing session data
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    with context.conversations_lock:
        conversation = context.conversations.get(session['session_id'], '')
    return render_template('chat.html', conversation=conversation)

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    assert 'session_id' in session
    user_message = request.form['message']
    if user_message is not None and user_message != '':
        # Don't let people send empty messages
        session_id = session['session_id']
        user_message = _preprocess_prompt(user_message)
        input_queue.put((session_id, user_message))
        with context.conversations_lock:
            context.conversations[session_id] = ''

        # Log user's prompt
        with open(f'server_logs/{session_id}.txt', 'a') as f:
            f.write('Prompt:\n' + user_message + '\n')
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})

@app.route('/stop_conversation', methods=['POST'])
def stop_conversation():
    assert 'session_id' in session
    # Don't let people send empty messages
    session_id = session['session_id']
    input_queue.put((session_id, '<|stop|>'))

    # Log user's prompt
    with open(f'server_logs/{session_id}.txt', 'a') as f:
        f.write('User requested stop\n')
    return jsonify({'success': True})

def respond_to_users():
    while True:
        response_session_id, response = output_queue.get()
        if response_session_id == INIT_ID:
            continue
        print(f'Got response {response} for session {response_session_id}')
        with context.conversations_lock:
            if response_session_id not in context.conversations:
                # User must have closed session, so we have nowhere to put this response.
                continue
            context.conversations[response_session_id] += response
            conversation = context.conversations[response_session_id]
        # Log response
        with open(f'server_logs/{response_session_id}.txt', 'a') as f:
            f.write(response)
        socketio.emit('new_data', conversation, room=response_session_id)

def poll_status():
    while True:
        prompt_q_size, context.num_decoding_users, decoding_users = status_queue.get()

        with context.conversations_lock:
            for user_id in context.conversations.keys():
                if user_id in decoding_users:
                    context.user_status[user_id] = 0
                else:
                    context.user_status[user_id] = prompt_q_size
                total_users = prompt_q_size + context.num_decoding_users
                print(f'Emit status for {user_id}')
                print('user_queue_pos: ', context.user_status[user_id])
                print('num_decoding_users: ', context.num_decoding_users)
                print('total_users', total_users)
                socketio.emit('new_status',
                            {'user_queue_pos': context.user_status[user_id],
                             'num_decoding_users': context.num_decoding_users,
                             'total_users': total_users},
                                room=user_id)



@socketio.on('connect')
def handle_connect():
    session_id = session.get('session_id')
    if session_id:
        join_room(session_id)
        print(f'Client connected and joined room {session_id}')
    else:
        print('No session_id found for connecting client')

@socketio.on('disconnect')
def handle_disconnect():
    session_id = session.get('session_id')
    if session_id:
        leave_room(session_id)
        with context.conversations_lock:
            if session_id in context.conversations:
                del context.conversations[session_id]
        print(f'Client with session {session_id} disconnected and left room')
    else:
        print('Client disconnected without session_id')

if __name__ == '__main__':
    # Create server log directory
    if not os.path.exists('server_logs'):
        os.makedirs('server_logs')
    initialize_decode_backend()
    socketio.run(app, debug=False, port=1223, host='0.0.0.0')
