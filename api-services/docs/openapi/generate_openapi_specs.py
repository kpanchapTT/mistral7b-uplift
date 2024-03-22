import json
from pathlib import Path


def get_llm_api_description(api_name):
    return f"# Tenstorrent {api_name}.\n\n## Request properties\n\n**text**: [required] prompt text sent to model to generate tokens.<br/>\n\n**temperature**: [optional, default=1.0] [value range: 0.01 to 100] For 0.1 < temperature < 1: amplifies certainty of model in its predictions before top_p sampling. For 1 < temperature < 10: reduces certainty of model in its predictions before top_p sampling.<br/>\n\n**max_tokens**: [optional, default=128] [value range: 1 to 2048] The maximum number of tokens to generate, fewer tokens may be generated and returned if the `stop_sequence` is reached before `max_tokens` tokens.<br/>\n\n**top_k**: [optional, default=40] [value range: 1 to 1000] Sampling option for next token selection, sample from most likely k tokens. Minimum value top_k=1 (greedy sampling) to maximum top_k=1000 (little effect).<br/>\n\n**top_p**: [optional, default=0.9] [value range: 0.01 to 1.0] Sampling option for next token selection, nucleus sampling from tokens within the the top_p value of cumulative probability. top_p=0.01 (near-greedy sampling) to top_p=1.0 (full multinomial sampling).<br/>\n\n**stop_sequence**: [optional, default=None] Stop generating tokens on first occurence of given sequence (can be a single character or sequence of characters). For example setting \".\" will stop at end of the first sentence, default `eos_token` defined by model tokenizer.<br/>\n\n## Response parameters\n\n**text**: model response to prompt as raw text sent using HTTP 1.1 chunked encoding as it is generated from the server backend. The response is truncated given request parameters `max_tokens` and `stop_sequence` as defined above.\n\n### Error responses parameters\n\n**status_code**: Error status code, e.g. 400.\n\n**message**: Description of error, e.g. Malformed JSON request body.\n\nIf response code is **Undocumented** with message 'TypeError: NetworkError when attempting to fetch resource.' it is likely a CORS issue. Requests to backend API made from browser must be made from the same subdomain, e.g. `app.tenstorrent.com`."


def get_llm_api_responses():
    return {
        "200": {
            "description": "Success, returns HTTP 1.1 chunked encoding to send raw text of each generated token to client."
        },
        "400": {
            "description": "Bad Request, returns `{'message': 'Helpful debug message for client.'}`, e.g. `Parameter: top_p is type=str, expected int`."
        },
        "401": {"description": "Unauthorized"},
        "500": {
            "description": "Internal Server Error, this is default for unhandled exceptions, as defined in backend server."
        },
        "503": {
            "description": "Service Unavailable, returns `{'message': 'Service overloaded, try again later.'}`"
        },
        # "Undocumented": {
        #     "description": "If details are: 'TypeError: NetworkError when attempting to fetch resource.' it is likely a CORS issue. Requests to backend API made from browser must be made from the same subdomain, e.g. `app.tenstorrent.com`."
        # },
    }


def get_llm_body():
    return {
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "top_k": {"type": "number"},
                        "top_p": {"type": "number"},
                        "temperature": {"type": "number"},
                        "max_tokens": {"type": "number"},
                        "stop_sequence": {"type": "string"},
                    },
                },
                "example": {
                    "text": "Translate 'hello world' into French.",
                    "top_k": 1,
                    "top_p": 0.9,
                    "temperature": 0.7,
                    "max_tokens": 32,
                    "stop_sequence": ".",
                },
            }
        }
    }


def get_summary(api_path):
    return f"{api_path} - POST"


def get_operationID(api_path):
    return f"post-{api_path.replace('/', '-')}".replace("--", "-")


def get_components():
    return {
        "securitySchemes": {
            "apiKeyHeader": {"type": "apiKey", "name": "Authorization", "in": "header"}
        }
    }


def get_security():
    return [{"apiKeyHeader": []}]


if __name__ == "__main__":
    environments = [
        "local",
        "dev",
        "prod",
    ]
    var_env_map = {
        "local": {
            "title": "LOCAL Tenstorrent Inference API",
            "servers": ["http://127.0.0.1:7000", "http://127.0.0.1:7001"],
            "apis": [
                {
                    "path": "/inference/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct Inference API",
                },
                {
                    "path": "/inference/falcon-7b-instruct",
                    "name": "Falcon 7B Instruct Inference API",
                },
                {
                    "path": "/conversation/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct Conversation API",
                },
            ],
        },
        "dev": {
            "title": "DEV Tenstorrent Inference API",
            "servers": ["https://app-dev.tenstorrent.com"],
            "apis": [
                {
                    "path": "/v1/key/chat/completions/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct Inference API",
                },
                {
                    "path": "/v1/key/chat/completions/falcon-7b-instruct",
                    "name": "Falcon 7B Instruct Inference API",
                },
                {
                    "path": "/v1/key/chat/conversation/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct conversation API",
                },
            ],
        },
        "prod": {
            "title": "Tenstorrent Inference API",
            "servers": ["https://app.tenstorrent.com"],
            "apis": [
                {
                    "path": "/v1/key/chat/completions/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct Inference API",
                },
                {
                    "path": "/v1/key/chat/completions/falcon-7b-instruct",
                    "name": "Falcon 7B Instruct Inference API",
                },
                {
                    "path": "/v1/key/chat/conversation/falcon-40b-instruct",
                    "name": "Falcon 40B Instruct conversation API",
                },
            ],
        },
    }

    for env in environments:
        openapi_json = {
            "openapi": "3.0.1",
            "info": {
                "title": var_env_map[env]["title"],
                "description": "Tenstorrent Inference API",
                "version": "1.0",
            },
            "servers": [{"url": url} for url in var_env_map[env]["servers"]],
            "paths": {
                api["path"]: {
                    "post": {
                        "summary": get_summary(api["path"]),
                        "operationId": get_operationID(api["path"]),
                        "description": get_llm_api_description(api["name"]),
                        "requestBody": get_llm_body(),
                        "responses": get_llm_api_responses(),
                    }
                }
                for api in var_env_map[env]["apis"]
            },
            "components": get_components(),
            "security": get_security(),
        }
        fpath = Path(__file__).parent / env / f"{env}_tenstorrent_llm_openapi_v3.json"
        with open(fpath, "w") as f:
            json.dump(openapi_json, f, indent=4)
