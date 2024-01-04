import multiprocessing
import os
import queue
import random
import shutil
import sys
import threading
import time
import uuid
from threading import Lock

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


# Shared variables with a lock for thread-safe access
context = Context()
time_last_response = time.time()
time_last_response_lock = Lock()


def get_falcon40b_backend_overrides(
    use_60_layers=True,
    use_2_layers=False,
    pytorch_no_weights=False,
    save_tti=False,
    load_tti=False,
    log_level_debug=False,
    tti_name="flash_decode_default_test.tti",
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
            os.path.join(inference_config.tti_cache, tti_name),
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
                os.path.join(inference_config.tti_cache, tti_name),
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
            os.path.join(inference_config.tti_cache, tti_name),
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
            "--enable-tvm-cache",
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
                os.path.join(inference_config.tti_cache, tti_name),
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
    tti_suffix = os.environ.get("FALCON_40B_TTI_SUFFIX", "v0")
    use_60_layers = not use_2_layers and not pytorch_no_weights
    tti_name = (
        f"flash_decode_60l_{tti_suffix}.tti"
        if use_60_layers
        else f"flash_decode_2l_{tti_suffix}_test.tti"
    )
    tti_path = os.path.join(inference_config.tti_cache, tti_name)
    print(
        f"getting overrides for:\n use_60_layers={use_60_layers},\n use_2_layers={use_2_layers},\n pytorch_no_weights={pytorch_no_weights},\n save_tti={save_tti},\n load_tti={load_tti},\n log_level_debug={log_level_debug},\n tti_name={tti_name}\n"
    )
    if save_tti:
        assert not os.path.exists(
            tti_path
        ), f"provided tti path exists: {tti_path}, cannot save over existing tti"
    elif load_tti:
        assert os.path.exists(
            tti_path
        ), f"provided tti path does not exist: {tti_path}, cannot load tti"
    assert not (
        pytorch_no_weights and save_tti
    ), "cannot save_tti with pytorch_no_weights."
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
        tti_name=tti_name,
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
        args=(
            input_queue,
            output_queue,
            status_queue,
            override_args,
            inference_config.backend_debug_mode,
        ),
    )
    p.start()
    default_params, _ = get_user_parameters({})
    input_queue.put((INIT_ID, "Dummy input for initialization", default_params))
    respond_to_users_thread = threading.Thread(target=respond_to_users)
    respond_to_users_thread.start()
    status_func_thread = threading.Thread(target=status_func)
    status_func_thread.start()


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


def _update_time_last_response():
    # only respond_to_users thread should update this value
    global time_last_response
    with time_last_response_lock:
        time_last_response = time.time()


def get_time_last_response():
    with time_last_response_lock:
        return time_last_response


def respond_to_users():
    MAX_USER_ROWS = 32
    while True:
        # q.get() will block the thread until output received
        response_session_id, response = output_queue.get()
        _update_time_last_response()
        if response_session_id == INIT_ID:
            continue
        if response_session_id not in output_queue_map:
            output_queue_map[response_session_id] = queue.Queue()
        output_queue_map[response_session_id].put(response)
        if inference_config.frontend_debug_mode:
            # Log response
            with open(f"server_logs/response_{response_session_id}.txt", "a") as f:
                f.write(response)
        # the outputs must be reclaimed
        _reclaim_output_queues()


def status_func():
    time_last_keep_alive_input = time.time()
    while True:
        time.sleep(1)
        # attempt to get backend status, skip if it is blocked waiting for input
        if not status_queue.empty():
            (
                prompt_q_size,
                num_decoding_users,
                decoding_users,
            ) = status_queue.get_nowait()
            print("num_decoding_users: ", num_decoding_users)
            print("prompt_q_size: ", prompt_q_size)
        time_since_response = time.time() - get_time_last_response()
        time_since_keep_live = time.time() - time_last_keep_alive_input
        if (
            time_since_response > inference_config.keepalive_input_period_seconds
            and time_since_keep_live > inference_config.keepalive_input_period_seconds
        ):
            session_id = "KEEP-ALIVE-INPUT"
            prompt = "the"
            params, _ = get_user_parameters(data={"max_tokens": 0})
            input_queue.put((session_id, prompt, params))
            time_last_keep_alive_input = time.time()
            print(
                f"keep alive: time_since_response={time_since_response}, time_since_keep_live={time_since_keep_live}"
            )


def preprocess_prompt(data):
    user_text, error = safe_convert_type(
        data_dict=data, key="text", dest_type=str, default=""
    )
    preprocessed_prompt = f"User: {user_text}\nAI: "
    return preprocessed_prompt, error


def safe_convert_type(data_dict, key, dest_type, default):
    error = None
    value = data_dict.get(key, default)
    converted_value = None
    try:
        converted_value = dest_type(value)
    # pylint: disable=broad-except
    except Exception as err:
        print(f"Error: safe_convert excepts: {err}")
        status_phrase = f"Parameter: {key} is type={type(value)}, expected {dest_type}"
        status_code = 400
        error = ({"message": status_phrase}, status_code)
    return converted_value, error


def apply_parameter_bounds(params):
    # clip parameters to within min / max boundaries
    error = None
    # (lower_bound, upper_bound)
    param_bounds = {
        "temperature": (0.01, 100.0),
        "top_p": (0.01, 1.0),
        "top_k": (1, 1000),
        "max_tokens": (1, 2048),
    }

    for key, (lower_bound, upper_bound) in param_bounds.items():
        value = params[key]
        within_bounds = lower_bound <= value <= upper_bound
        if not within_bounds:
            status_phrase = f"Parameter: {key} is outside bounds, {lower_bound} <= {value} <= {upper_bound}."
            status_code = 400
            error = ({"message": status_phrase}, status_code)
            return {}, error
    return params, error


def get_user_parameters(data):
    """This function turns user input into parameters."""
    # (default_value, python_type)
    default_params = {
        "temperature": (1.0, float),
        "top_p": (0.9, float),
        "top_k": (10, int),
        "max_tokens": (128, int),
        "stop_sequence": (None, str),
        "return_prompt": (False, bool),
    }
    error = None
    params = {}
    # user input sanitization to expected python types, or default values, with error handling
    for key, (default_value, python_type) in default_params.items():
        value, error = safe_convert_type(
            data_dict=data, key=key, dest_type=python_type, default=default_value
        )
        if error is not None:
            # return 400 to user on first error
            return {}, error
        params[key] = value

    return params, error


def sanitize_request(request):
    error = None
    user_session_id = None

    if request.is_json:
        data = request.get_json()
    else:
        error = {"message": "Request was not JSON"}, 400
        return None, None, None, error

    prompt, error = preprocess_prompt(data)
    if error:
        return None, None, None, error

    if not prompt:
        error = {
            "message": "required 'text' parameter is either empty or not provided"
        }, 400
        return None, None, None, error

    params, error = get_user_parameters(data)
    if error:
        return None, None, None, error

    params, error = apply_parameter_bounds(params)
    if error:
        return None, None, None, error

    if "session_id" in data:
        user_session_id, error = safe_convert_type(data, "session_id", str, None)
        if error:
            return None, None, None, error

    return prompt, params, user_session_id, error


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
            time.sleep(0.02)
            continue
        elif session_id not in output_queue_map and started_generation:
            # generation ended without EOS token
            print(f"session_id: {session_id} ended without EOS.")
            done_generation = True
            continue

        # use nowait and continue sleep loop to avoid reading from q after slot_idx reallocated
        if output_queue_map[session_id].empty():
            time.sleep(0.02)
            continue

        out_text = output_queue_map[session_id].get_nowait()
        if out_text.endswith("<|endoftext|>"):
            done_generation = True
            with context.conversations_lock:
                del context.user_last_read[session_id]

        if inference_config.frontend_debug_mode:
            with open(f"server_logs/user_{session_id}.txt", "a") as f:
                f.write(out_text)

        yield out_text


@app.route("/predictions/falcon40b", methods=["POST"])
def inference():
    start_time = time.time()
    # user will get 400 on invalid input, with helpful status message
    prompt, params, user_session_id, error = sanitize_request(request)
    if error:
        return error

    # create a session_id if not supplied
    if "session_id" not in session and user_session_id is None:
        session["session_id"] = str(uuid.uuid4())
    else:
        print(f"PRE-EXISTING SESSION: {session.get('session_id')}, {user_session_id}")
        # TODO: add user_session_id as session_id if passed correctly
        # currently only support stateless sessions
        session["session_id"] = str(uuid.uuid4())

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
        return {"message": "Service busy"}, 500

    # input
    session_id = session.get("session_id")
    input_queue.put((session_id, prompt, params))

    if inference_config.frontend_debug_mode:
        # Log user's prompt
        with open(f"server_logs/prompt_{session_id}.txt", "a") as f:
            f.write("Prompt:\n" + prompt + "\n")

    # output
    return Response(get_output(session_id), content_type="text/event-stream")


def create_server():
    # Create server log directory
    if not os.path.exists("server_logs"):
        os.makedirs("server_logs")
    override_args = get_backend_override_args()
    initialize_decode_backend(override_args)
    return app


if __name__ == "__main__":
    app = create_server()
    app.run(debug=False, port=inference_config.backend_server_port, host="0.0.0.0")
