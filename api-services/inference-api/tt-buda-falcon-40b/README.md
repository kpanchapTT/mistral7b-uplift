# TT Buda Falcon 40B Infereience API

## Development notes

To run locally at maximum iteration speed:

1. use `docker compose up` with `docker-compose.yml` to create the service.
2. use `docker exec` to run mock inference API server (see section below)


### Run with server code changes faster

#### Using Flask debug mode and hot reloading

When using Docker mounted volumes in `/mnt` the hot reloading appears to take a very long time.
The message ` * Detected change in '/mnt/tt-buda-falcon-40b/src/inference_api_server.py', reloading` can be seen
when the feature is configured by 1) setting env var `FLASK_ENV=development`, 2) 
running the mock inference server with:
```python
    app.run(
        port=inference_config.backend_server_port,
        host="0.0.0.0",
        debug=True,
        use_reloader=True,
    )
```

For this reason, it is faster in practice to use docker exec into the container and run the service.

#### Using docker compose

To run the service foregrounded for python interactive debugging with pdf breakpoints use the docker-compose.yml command: `command: ["/bin/bash", "-c", "sleep infinity"]`

```bash
docker compose up

export IMAGE_NAME='project-falcon/falcon40b-demo:v0.0.17-local'
docker exec -it $(docker ps | grep "${IMAGE_NAME}" | awk '{print $1}') bash

cd /mnt/src
gunicorn --config gunicorn.conf.py
```

### JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_ENCODED=$(/mnt/scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export JWT_TOKEN="Bearer ${JWT_ENCODED}"
```

__Note:__ when using `curl` as bellow make sure to have double quotes `"` around "Authorization: $JWT_TOKEN", this allows for the shell defined variables to be expanded, otherwise if using single quotes `'` the string literal `$JWT_TOKEN` will be sent and authorization will fail.

## testing

```bash
curl -X 'POST' \
  'http://127.0.0.1:7000/inference/falcon40b' \
  -b cookies.txt \
  -c cookies.txt \
  -L \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -H "Authorization: $JWT_TOKEN" \
  -d '{
  "text": "What number did you repeat?",
  "top_k": 1,
  "top_p": 0.9,
  "temperature": 0.7,
  "max_tokens": 32,
  "stop_sequence": "."
}'
```

## Run tests with mock model

use docker compose as instructed above. For testing of just the flask server a mock_backend is available:
```bash
# run in separate terminals or processes
python src/test_inference_api.py
python src/test_decode_backend_v1.py
```

#### Run model tests

This requires setting up a Galaxy server for deployment then running the test scripts:
```bash
python src/test_decoding.py
# this requires a running inference server as described above
python src/test_inference_api.py
```

The 1L pytorch version can also be run as a test application:
```bash
export FALCON_40B_PYTORCH_NO_WEIGHTS='1'
python src/inference_api_server.py
```


#### Using docker exec to run

This is not the preferred way to debug, if possible use `docker compose` as mentioned above.

```bash
# get the container ID of the running docker container of the image
export IMAGE_NAME='project-falcon/falcon40b-demo:v0.0.17-local'
docker exec -it $(docker ps | grep "${IMAGE_NAME}" | awk '{print $1}') bash
```

Kill the inference API server that was started by container, run your own modified version:
```bash 
# kill with SIGINT PID of process running on 7000 (inference API server), can verify process is terminated with `ps -e`
sudo apt update && sudo apt install lsof
kill -15 $(lsof -i :7000 | awk 'NR>1 {print $2}')
gunicorn --config gunicorn.conf.py
```

# Documentation 

Flask API server frontend documentation: [../../docs/tt-buda-falcon-40b/frontend.md](../../docs/tt-buda-falcon-40b/frontend.md)
Galaxy Falcon 40B backend documentation: [../../docs/tt-buda-falcon-40b/backend.md](../../docs/tt-buda-falcon-40b/backend.md)

## Life cycle of requests

Each new request is handled on a separate thread. The prompt is validated then pushed onto the input_queue.

The backend aysnchronously processes input_queue prompts into available user_rows in the decoding. Currently a fixed 32 user_rows is supported. The prefill of the on device KV cache (ODKV) is done by the same graph on the TT device.

Each token is pushed onto the output_queue once it is generated. The output_queue is read aysnchronously by a single threaded process within the flask server and stored in a dict of queues keyed on the session_id (or user_id), this means no two users can get eachother's response tokens. Each token is read by the request processes from this dict using the corresponding session_id and streamed back to the user using HTTP 1.1 chunked encoding.

When decoding is finished for a user within the user_rows it is evicted from the `DecodeBackend.users` list. On the next loop over inputing data from the input_queue this user_row will be filled if there is input prompt data for it.

The flask server request process having received the EOS token marks the connection closed and it's session_id as ready for deletion in the dict of session_id keyed queues.

## API parameters

**text**: [required] prompt text sent to model to generate tokens.<br/>

**temperature**: [optional] <br/>

  0.1 < temperature < 1: amplifies certainty of model in its predictions before top_p sampling.<br/>
  1 < temperature < 10: reduces certainty of model in its predictions before top_p sampling.<br/>

**max_tokens**: [optional] [1 <= max_tokens <=2048] the maximum number of tokens to generate, fewer tokens may be generated and returned if the `stop_sequence` is reached before `max_tokens` tokens.<br/>

**top_k**: [optional] sampling option for next token selection, sample from most likely k tokens. Minimum value top_k=1 (greedy sampling) to maximum top_k=1000 (little effect).<br/>

**top_p**: [optional] sampling option for next token selection, nucleus sampling from tokens within the the top_p value of cumulative probability. top_p=0.01 (near-greedy sampling) to top_p=1.0 (full multinomial sampling).<br/>

**stop_sequence**: [optional] stop generating tokens on first occurence of given sequence (can be a single character or sequence of characters). Options e.g. "." stop at sentance or default is `eos_token` or `"<|endoftext|>"`.<br/>

### Response parameters

**text**: model response to prompt as raw text sent using HTTP 1.1 chunked uncoding as it is generated from the server backend. The response is truncated given request parameters `max_tokens` and `stop_sequence` as defined above.

##### Error responses parameters

**status_code**: Error status code, e.g. 400.

**message**: Description of error, e.g. Malformed JSON request body.

### example requests

```bash
curl ${LLM_CHAT_API_URL} -H "Content-Type: application/json" \
-H "Authorization: ${AUTHORIZATION}" \
-d '{"text":"Have you been to Paris, France?"}'

# {"answer":"Have you been to Paris, France? Then come to Paris, Michigan!
# \nParis, Mich., is a community of 7,000 located in northeast Kent County"}

curl ${LLM_CHAT_API_URL} -H "Content-Type: application/json" \
-H "Authorization: ${AUTHORIZATION}" \
-d '{"text":"Have you been to Paris, France?",
  temperature":"1",
  "top_k":"40",
  "top_p":"0.9",
  "max_tokens":"32"
}'

# {"answer":"Have you been to Paris, France? If you have, or if you\ haven\u2019t,\
# you are sure to have heard of a lot of interesting things the French"}

curl ${LLM_CHAT_API_URL} -H "Content-Type: application/json" \
-H "Authorization: ${AUTHORIZATION}" \
-d '{"text":"Have you been to Paris, France?", 
  "temperature":"1", 
  "top_k":"40",
  "top_p":"0.9",
  "max_tokens":"32",
  "stop_sequence": ".",
}'

# {"answer":"Have you been to Paris, France? Do you know the Eiffel tower? \
# Did you have the opportunity to visit it?\nI had the opportunity to visit the"}
```

# clean tt artifacts

```bash
rm -rf .hlkc_cache/ .pkl_memoize_py3/ generated_modules/ tt_build/
```

