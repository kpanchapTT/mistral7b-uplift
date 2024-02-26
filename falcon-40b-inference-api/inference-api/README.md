## 

## Development notes

To run locally at maximum iteration speed:

1. use `docker compose up` with `docker-compose.yml` to create the service.
2. use `docker exec` to run mock inference API server (see section below)


### Run with server code changes faster

#### Using Flask debug mode and hot reloading

When using Docker mounted volumes in `/mnt` the hot reloading appears to take a very long time.
The message ` * Detected change in '/mnt/inference-api/inference_api_server.py', reloading` can be seen
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

#### Using docker exec to run

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
python -u /mnt/src/_mock_inference_api_server.py
```
