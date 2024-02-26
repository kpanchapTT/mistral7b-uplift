FROM alpine:latest

# Install Python.
# If you need a specific Python version, specify python3=<version>.
RUN apk update && \
    apk add --no-cache --update python3 py3-pip curl
    # && apk add --no-cache --virtual .build-deps gcc python3-dev musl-dev libffi-dev \
    # && apk add --no-cache g++ make gfortran openblas-dev

# Set Python as the default.
# RUN ln -s /usr/bin/python3 /usr/bin/python

ARG APP_DIR=/app
ARG HOME_DIR=/home/user

# Copy your application code to the container
COPY ./src ${APP_DIR}/src
COPY ./requirements.txt ${APP_DIR}/requirements.txt
COPY ./run_conversation_api.sh ${APP_DIR}/run_conversation_api.sh

# Install any needed packages specified in requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /app/requirements.txt

# Set up work directory
WORKDIR ${APP_DIR}

CMD sleep infinity