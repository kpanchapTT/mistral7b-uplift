# Setup docker compose environment

## .env file

the `.env` is picked up by `docker compose up` command and uses variable expansion within `docker-compose.yml`. Make sure `.env` is not commited, however it is not used in production, only for local testing so no secure secret management is required.

example .env:
```
JWT_SECRET=example-test-secret-456
FLASK_SECRET=example-flask-secret-8904873
JWT_TOKEN=Bearer <get result from below using your JWT_SECRET>
```

## JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_ENCODED=$(/mnt/scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export JWT_TOKEN="Bearer ${JWT_ENCODED}"
```

## docker-compose.yml

This file defines the local services and networking between them.
- ports are only needed for external access (e.g. swagger-ui)
- mounting volumes is added for rapid iteration and logs viewing
-

## CORS handling

CORS is allowed using the TEST_SERVER env var in `conversation-api`, and MOCK_MODEL for `inference-api`. This allows swagger-ui to work when running locally in the browser.
CORS is not allowed in production currently, if required allowed origins will be added.
