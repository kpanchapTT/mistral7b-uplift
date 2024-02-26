# Conversation API

Flask server to add a minimal conversation memory layer to stateless LLM inference API.


## run via docker compose

```
cd project-falcon/falcon-40b-inference-api
docker compose up
```

## environment vars:

TT_API_KEY
TT_API_URL
FLASK_SECRET

###

```bash
# get the container ID of the running docker container of the image
export IMAGE_NAME='project-falcon/conversation-api:v0.0.1'
docker exec -it $(docker ps | grep "${IMAGE_NAME}" | awk '{print $1}') sh
```



###
```bash
pip install --break-system-packages flask-cors black pyjwt
cd /mnt
python src/conversation_api_server.py
```

Test conversation API with storing and sending cookies in `cookies.txt`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:7001/conversation/falcon-40b-instruct' \
  -b cookies.txt \
  -c cookies.txt \
  -L \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Repeat the number '\''94572'\''.",
  "top_k": 1,
  "top_p": 0.9,
  "temperature": 0.7,
  "max_tokens": 32,
  "stop_sequence": "."
}'

curl -X 'POST' \
  'http://127.0.0.1:7001/conversation/falcon-40b-instruct' \
  -b cookies.txt \
  -c cookies.txt \
  -L \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "What number did you repeat?",
  "top_k": 1,
  "top_p": 0.9,
  "temperature": 0.7,
  "max_tokens": 32,
  "stop_sequence": "."
}'
```

## design

Use alpine Docker image as base with minimal requirements for running a flask server.

In memory datastructure from hatstack (https://github.com/deepset-ai/haystack) to handle conversation memory and make calls to inference API. The source code is directly used (Apache 2.0) rather than importing the entire library to keep the Docker image minimal.

Streaming is handled efficiently by only writing to conversation memory on completion of the response stream.



