FROM alpine:latest

# Install Python.
RUN apk update && \
    apk add --no-cache --update python3 py3-pip curl

ARG APP_DIR=/falcon-40b-conversation-api
# ARG HOME_DIR=/home/user

# Copy application code to the container
COPY ./src ${APP_DIR}/src
COPY ./requirements.txt ${APP_DIR}/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r ${APP_DIR}/requirements.txt

# Set up work directory
WORKDIR ${APP_DIR}/src

CMD ["gunicorn", "--config", "gunicorn.conf.py"]

# default port is 7001
ENV SERVICE_PORT=7001  
HEALTHCHECK --retries=5 --start-period=1s CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
