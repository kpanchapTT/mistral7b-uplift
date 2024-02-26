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
cd /mnt
python src/conversation_api_server.py
```









## design

Use alpine Docker image as base with minimal requirements for running a flask server.

In memory datastructure from hatstack (https://github.com/deepset-ai/haystack) to handle conversation memory and make calls to inference API. The source code is directly used (Apache 2.0) rather than importing the entire library to keep the Docker image minimal.

Streaming is handled efficiently by only writing to conversation memory on completion of the response stream.



