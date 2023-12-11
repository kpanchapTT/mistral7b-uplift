import multiprocessing
import os
import queue
import sys
import threading
import time
import uuid
from threading import Lock
import random
import shutil

from flask import Flask, Response, jsonify, request, session

sys.path.append(os.getcwd())

from decode_backend_v1 import run_decode_backend
from inference_config import inference_config

app = Flask(__name__)
app.secret_key = "your_secret_key"
INIT_ID = "COMPILE-INITIALIZATION"


class Context:
    # Store current context
    # Store conversation history
    # Initialize the lock
    def __init__(self):
        self.conversations = {}
        self.user_status = {}  # {user_id:q_position}
        self.num_decoding_users = 0
        self.user_last_read = {}
        self.user_parameters = {}
        # Initialize the lock
        self.conversations_lock = Lock()


context = Context()


def get_falcon40b_backend_overrides(
    use_60_layers=True,
    use_2_layers=False,
    pytorch_no_weights=False,
    save_tti=False,
    load_tti=False,
    log_level_debug=False,
):
    log_level = "ERROR"
    if log_level_debug:
        log_level = "DEBUG"
    # 2 layer model is used for debugging and testing
    if use_2_layers and save_tti:
        copy_tvm_cache_to_cwd()
        override_args = [
            "--mode",
            "concurrent",
            "-l",
            "2",  # 2 layers
            "--version",
            "efficient-40b",
            "-d",
            "silicon",
            "--arch",
            "nebula-galaxy",
            "--num-tokens",
            "20",
            "--user-rows",
            "32",
            "--precision",
            "bf16",
            "--num-chips",
            "32",
            "-mf",
            "8",
            "--log-level",
            "DEBUG",
            "--opt-level",
            "4",
            "--hf-cache",
            inference_config.hf_cache,
            "-odlmh",
            "-plmh",
            "-fv",
            "--flash-decode",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--save",
            inference_config.tti_cache + "/flash_decode_2l_v0_test.tti",
            "--load-pretrained",
            "--model",
            "tiiuae/falcon-40b-instruct",
        ]
    elif use_2_layers and not save_tti:
        override_args = [
            "--mode",
            "concurrent",
            "-l",
            "2",  # 2 layers
            "--version",
            "efficient-40b",
            "-d",
            "silicon",
            "--arch",
            "nebula-galaxy",
            "--num-tokens",
            "20",
            "--num-outer-loops",
            "1",  # default value
            "--user-rows",
            "32",
            "--precision",
            "bf16",
            "--num-chips",
            "32",
            "-mf",
            "8",
            "--log-level",
            log_level,
            "--opt-level",
            "4",
            "--hf-cache",
            inference_config.hf_cache,
            "-odlmh",
            "-plmh",
            "-fv",
            "--flash-decode",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--load-pretrained",
            "--model",
            "tiiuae/falcon-40b-instruct",
        ]
        if load_tti:
            override_args += [
                "--load",
                inference_config.tti_cache + "/flash_decode_2l_v0_test.tti",
            ]
    elif use_60_layers and save_tti:
        copy_tvm_cache_to_cwd()
        override_args = [
            "--mode",
            "sequential",
            "-l",
            "60",
            "--version",
            "efficient-40b",
            "-d",
            "silicon",
            "--arch",
            "nebula-galaxy",
            "--num-tokens",
            "5000",
            "--user-rows",
            "32",
            "--precision",
            "bf16",
            "--num-chips",
            "32",
            "-mf",
            "8",
            "--log-level",
            log_level,
            "--opt-level",
            "4",
            "--hf-cache",
            inference_config.hf_cache,
            "--enable-tvm-cache",
            "-odlmh",
            "-plmh",
            "-fv",
            "--flash-decode",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--save",
            inference_config.tti_cache + "/flash_decode_60l_v0_instruct.tti",
            "--load-pretrained",
            "--model",
            "tiiuae/falcon-40b-instruct",
        ]
    elif use_60_layers and not save_tti:
        override_args = [
            "--mode",
            "concurrent",
            "-l",
            "60",
            "--version",
            "efficient-40b",
            "-d",
            "silicon",
            "--arch",
            "nebula-galaxy",
            "--num-tokens",
            "1000000",
            "--num-outer-loops",
            "1000",
            "--user-rows",
            "32",
            "--precision",
            "bf16",
            "--num-chips",
            "32",
            "-mf",
            "8",
            "--log-level",
            log_level,
            "--opt-level",
            "4",
            "--hf-cache",
            inference_config.hf_cache,
            "-odlmh",
            "-plmh",
            "-fv",
            "--flash-decode",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--load-pretrained",
            "--model",
            "tiiuae/falcon-40b-instruct",
        ]
        if load_tti:
            override_args += [
                "--load",
                inference_config.tti_cache + "/flash_decode_60l_v0_instruct.tti",
            ]
    elif pytorch_no_weights:
        # 1L pytorch no weights for debug and testing
        override_args = [
            "--mode",
            "concurrent",
            "-l",
            "1",
            "--version",
            "efficient-40b",
            "-d",
            "pytorch",  # pytorch instead of silicon
            "--arch",
            "nebula-galaxy",
            "--num-tokens",
            "1_000_000_000",
            "--user-rows",
            "32",
            "--precision",
            "fp32",
            "--num-chips",
            "32",
            "-mf",
            "8",
            "--log-level",
            "DEBUG",
            "--opt-level",
            "4",
            "--hf-cache",
            inference_config.hf_cache,
            "-odlmh",
            "-plmh",
            "-fv",
            "--flash-decode",
            "--top-k",
            "5",
            "--top-p",
            "0.9",
            "--model",
            "tiiuae/falcon-40b-instruct",
        ]
    else:
        raise ValueError(
            f"Invalid overrides for: use_60_layers={use_60_layers}, use_2_layers={use_2_layers}, pytorch_no_weights={pytorch_no_weights}, save_tti={save_tti}, load_tti={load_tti}"
        )
    return override_args


def get_backend_override_args():
    # terminate env vars and pass to switching logic with simple logging
    use_2_layers = os.environ.get("FALCON_40B_2LAYER") == "1"
    pytorch_no_weights = os.environ.get("FALCON_40B_PYTORCH_NO_WEIGHTS") == "1"
    save_tti = os.environ.get("FALCON_40B_SAVE") == "1"
    load_tti = os.environ.get("FALCON_40B_LOAD") == "1"
    log_level_debug = os.environ.get("FALCON_40B_LOG_LEVEL_DEBUG") == "1"
    use_60_layers = not use_2_layers and not pytorch_no_weights
    assert not (
        pytorch_no_weights and save_tti
    ), "cannot save_tti with pytorch_no_weights."
    print(
        f"getting overrides for: use_60_layers={use_60_layers}, use_2_layers={use_2_layers}, pytorch_no_weights={pytorch_no_weights}, save_tti={save_tti}, load_tti={load_tti}, log_level_debug={log_level_debug}"
    )
    if pytorch_no_weights or use_2_layers:
        print(
            f"WARNING: pytorch_no_weights={pytorch_no_weights}, use_2_layers={use_2_layers} is run for debug and testing only."
        )
    override_args = get_falcon40b_backend_overrides(
        use_60_layers=use_60_layers,
        use_2_layers=use_2_layers,
        pytorch_no_weights=pytorch_no_weights,
        save_tti=save_tti,
        load_tti=load_tti,
        log_level_debug=log_level_debug,
    )
    print(override_args)
    return override_args


def copy_tvm_cache_to_cwd():
    # backend tvm cache is assumed to be in the calling cwd
    current_dir = os.getcwd()
    if os.path.isdir(inference_config.tvm_cache):
        files = os.listdir(inference_config.tvm_cache)
        # Iterate over the files
        for file_name in files:
            src_file = os.path.join(inference_config.tvm_cache, file_name)
            if os.path.isfile(src_file) and file_name.startswith("tvm"):
                # Copy the file to the current working directory
                dest_file = os.path.join(os.getcwd(), file_name)
                shutil.copy(src_file, dest_file)


verbose = False


def initialize_decode_backend(override_args):
    global input_queue
    global output_queue
    global status_queue
    global output_queue_map
    output_queue_map = {}
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    status_queue = multiprocessing.Queue()
    # run the decode backend in a separate process
    p = multiprocessing.Process(
        target=run_decode_backend,
        args=(input_queue, output_queue, status_queue, override_args, verbose),
    )
    p.start()
    input_queue.put(
        (INIT_ID, "Dummy input for initialization", get_user_parameters({}))
    )
    respond_to_users_thread = threading.Thread(target=respond_to_users)
    respond_to_users_thread.start()
    poll_status_thread = threading.Thread(target=poll_status)
    poll_status_thread.start()


def _reclaim_output_queues():
    """reclaim resources for output queues for user_ids that are:
    1. not in self.users (have completed generation)
    2. are empty (have been read out by request handling thread)

    Only this function deletes from the output_queue_map in a single thread.
    """
    current_time = time.time()

    active_user_ids = {
        user_id
        for user_id, last_read_time in context.user_last_read.items()
        if current_time - last_read_time < inference_config.max_inactive_seconds
    }
    marked_for_deletion = set()
    for user_id, output_q in output_queue_map.items():
        if user_id not in active_user_ids and output_q.empty():
            marked_for_deletion.add(user_id)

    for user_id in marked_for_deletion:
        del output_queue_map[user_id]


def respond_to_users():
    MAX_USER_ROWS = 32
    loop = 0
    while True:
        response_session_id, response = output_queue.get()
        if response_session_id == INIT_ID:
            continue
        if response_session_id not in output_queue_map:
            output_queue_map[response_session_id] = queue.Queue()
        output_queue_map[response_session_id].put(response)
        # Log response
        with open(f"server_logs/response_{response_session_id}.txt", "a") as f:
            f.write(response)
        loop += 1
        if loop % MAX_USER_ROWS == 0:
            _reclaim_output_queues()


def poll_status():
    while True:
        time.sleep(1)
        prompt_q_size, num_decoding_users, decoding_users = status_queue.get()
        print("num_decoding_users: ", num_decoding_users)
        print("prompt_q_size: ", prompt_q_size)


def validate_request(request):
    error = None
    if request.is_json:
        data = request.get_json()
    else:
        error = "Request was not JSON", 400
        return None, error

    if not data.get("text"):
        error = "required 'text' parameter is either empty or not given", 400
    return data, error


def get_user_parameters(data):
    default_temperature = 1.0
    default_top_p = 0.9
    default_top_k = 10
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
    started_generation = False
    while not done_generation:
        if session_id in output_queue_map and not started_generation:
            started_generation = True
            with context.conversations_lock:
                context.user_last_read[session_id] = time.time()
        elif session_id not in output_queue_map and not started_generation:
            # waiting for start of generation
            time.sleep(0.03)
            continue
        elif session_id not in output_queue_map and started_generation:
            # generation ended without EOS token
            print(f"session_id: {session_id} ended without EOS.")
            done_generation = True
            continue

        # use nowait and continue sleep loop to avoid reading from q after slot_idx reallocated
        if output_queue_map[session_id].empty():
            time.sleep(0.03)
            continue

        out_text = output_queue_map[session_id].get_nowait()
        if out_text == "<|endoftext|>":
            done_generation = True
            with context.conversations_lock:
                del context.user_last_read[session_id]
        # Log response
        with open(f"server_logs/user_{session_id}.txt", "a") as f:
            f.write(out_text)

        yield out_text


@app.route("/predictions/falcon40b", methods=["POST"])
def inference():
    start_time = time.time()
    data, error = validate_request(request)
    if error:
        return error

    # create a session_id if not supplied
    if "session_id" not in session and "session_id" not in data:
        session["session_id"] = str(uuid.uuid4())
    else:
        print(f"PREVIOUS EXISTING SESSION: {session['session_id']}")

    # if input_q full, retry with simple back-off
    for timeout in [1, 1, 1, 1, 1, 5, 5, 5, 10]:
        if input_queue.qsize() >= inference_config.max_input_qsize:
            # add jitter
            sleep_t = timeout * random.random()
            print(f"retry: {sleep_t}, session: {session['session_id']} ")
            time.sleep(sleep_t)
        else:
            break
    else:
        return "Service busy", 500

    # input
    session_id = session.get("session_id")
    user_message = data["text"]
    user_message = _preprocess_prompt(user_message)
    params = get_user_parameters(data)
    input_queue.put((session_id, user_message, params))

    # Log user's prompt
    with open(f"server_logs/prompt_{session_id}.txt", "a") as f:
        f.write("Prompt:\n" + user_message + "\n")

    # output
    return Response(get_output(session_id), content_type="text/event-stream")


def _preprocess_prompt(prompt):
    preprocessed_prompt = f"User: {prompt}\nAI:"
    return preprocessed_prompt


if __name__ == "__main__":
    # Create server log directory
    if not os.path.exists("server_logs"):
        os.makedirs("server_logs")
    override_args = get_backend_override_args()
    breakpoint()
    # initialize_decode_backend(override_args)
    # app.run(debug=False, port=inference_config.backend_server_port, host="0.0.0.0")
