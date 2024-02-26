from conversation_config import conversation_config

workers = 1
bind = f"127.0.0.1:{conversation_config.backend_server_port}"
reload = False
wsgi_app = "conversation_api_server:create_server()"
worker_class = "gthread"
threads = 96
timeout = 120
