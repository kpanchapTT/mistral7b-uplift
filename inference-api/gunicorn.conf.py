from inference_config import inference_config

workers = 1
bind = f"127.0.0.1:{inference_config.backend_server_port}"
reload = False
wsgi_app = "inference_api_server:create_server()"
worker_class = "gthread"
threads = 96
