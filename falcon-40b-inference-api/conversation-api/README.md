# Conversation API

Flask server to add a minimal conversation memory layer to stateless LLM inference API.


### JWT_TOKEN Authorization
To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_ENCODED=$(/mnt/scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export JWT_TOKEN="Bearer ${JWT_ENCODED}"
```

## run via docker compose

To run the service foregrounded for python interactive debugging with pdf breakpoints use the docker-compose.yml command: `command: ["/bin/bash", "-c", "sleep infinity"]`

```bash
# define JWT_TOKEN for docker-compose.yml to pick up
export JWT_TOKEN="Bear ${JWT_ENCODED}"
docker compose up

export IMAGE_NAME='project-falcon/conversation-api:v0.0.1'
docker exec -it $(docker ps | grep "${IMAGE_NAME}" | awk '{print $1}') sh

cd /mnt/src
gunicorn --config gunicorn.conf.py
```

## environment vars:

Environment variables except for secrets are defined in `conversation_config.py`.

### Secrets
TT_API_JWT: the JWT_TOKEN inference API is using for authorization
FLASK_SECRET: used by flask server for session cookie signing, Cross-Site Request Forgery (CSRF) protection, and any cryptographicly secured operation.
JWT_SECRET: used for authorization of the servers routes

###
```bash
# pip install --break-system-packages flask-cors black pyjwt
cd /mnt/src
gunicorn --config gunicorn.conf.py

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
  -H "Authorization: ${JWT_TOKEN}" \
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



