import pathlib
from datetime import datetime

from conversation_config import conversation_config

workers = 1
bind = f"0.0.0.0:{conversation_config.backend_server_port}"
reload = False
wsgi_app = "conversation_api_server:create_server()"
worker_class = "gthread"
threads = 96
timeout = 120

# set log files
if not pathlib.Path(conversation_config.log_cache).exists():
    pathlib.Path(conversation_config.log_cache).mkdir(parents=True, exist_ok=True)
datetime_prefix = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
accesslog = f"{conversation_config.log_cache}/{datetime_prefix}_access.log"
errorlog = f"{conversation_config.log_cache}/{datetime_prefix}_error.log"
loglevel = "info"


# switch between mock model and production version
if conversation_config.test_server:
    wsgi_app = "conversation_api_server:create_test_server()"
else:
    wsgi_app = "conversation_api_server:create_server()"
