cd /home/ubuntu/project-falcon/api-services
docker run --rm -it \
    --device /dev/tenstorrent/0:/dev/tenstorrent/0 \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    --shm-size=4g \
    --cap-add ALL \
    -v ./inference-api/tt-metal-falcon-7b/src:/tmp/src \
    -v /home/ubuntu/project-falcon/local_cache_root/:/tmp/local_cache_root/ \
    -e JWT_SECRET=$(JWT_SECRET) \
    -e FLASK_SECRET=$(FLASK_SECRET) \
    -e CACHE_ROOT=/tmp/local_cache_root \
    -e SERVICE_PORT=7000 \
    mistral-image bash
