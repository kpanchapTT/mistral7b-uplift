# Build stage
FROM ubuntu:20.04 as builder
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"

ENV SHELL=/bin/bash

USER root
WORKDIR /tmp/install

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

RUN apt update && apt install -y \
    ## TT-Metalium Backend Dependencies: https://github.com/tenstorrent-metal/tt-metal/blob/main/INSTALLING.md#installing-dependencies-on-ubuntu-before-accelerator-level
    software-properties-common=0.99.9.12 \
    build-essential=12.8ubuntu1.1 \
    python3.8-venv=3.8.10-0ubuntu1~20.04.9 \
    libgoogle-glog-dev=0.4.0-1build1 \
    libyaml-cpp-dev=0.6.2-4ubuntu1 \
    libboost-all-dev=1.71.0.0ubuntu2 \
    libsndfile1=1.0.28-7ubuntu0.2 \
    libhwloc-dev \
    graphviz \
    # build deps
    tzdata \
    clang-6.0=1:6.0.1-14 \
    python-is-python3 \
    pciutils \
    python3-pip \
    git \
    git-lfs \
    # tt-metal dev deps
    pandoc \
    libtbb-dev \
    libcapstone-dev \
    pkg-config \
    # user tools
    htop \
    nano \
    screen \
    tmux \
    vim \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

## add user
ARG APP_DIR=/tt-metal-mistral-7b
ARG HOME_DIR=/home/user

RUN useradd -u 1000 -s /bin/bash -d ${HOME_DIR} user && \
    mkdir -p ${HOME_DIR} && \
    chown -R user:user ${HOME_DIR}

USER user
WORKDIR ${HOME_DIR}

## build tt-metal
ENV PATH=$PATH:${HOME_DIR}/.local/bin
# ENV ARCH_NAME=grayskull
ENV ARCH_NAME=wormhole_b0
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV TT_METAL_ENV=dev
ENV PYTHONPATH=${HOME_DIR}/tt-metal
# NOTE: production build is not documented
# ENV TT_METAL_ENV=production
# ENV TT_METAL_CREATE_STATIC_LIB=1
# default port is 7000
ENV SERVICE_PORT=7000

# clone tt-metal and its submodules
# remove git data to reduce space post-build
# May 24th, 2024, rc: main -> 35236b6532f85b1b078b32318d0bf3686298cbae
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git && \
    cd tt-metal && \
    git checkout 35236b6532f85b1b078b32318d0bf3686298cbae && \ 
    git submodule update --init --recursive && \
    git submodule foreach 'git lfs fetch --all && git lfs pull' && \
    pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    make build && \
    rm -rf .git

RUN git clone https://github.com/tenstorrent/tt-smi.git && \
    cd tt-smi && \ 
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    python3 -m venv .venv && \
    . .venv/bin/activate && \ 
    pip3 install --upgrade pip && \
    . "$HOME/.cargo/env" && \
    pip3 install .


# install app requirements
WORKDIR "${HOME_DIR}/${APP_DIR}"
COPY --chown=user:user "src" "${HOME_DIR}/${APP_DIR}/src"
COPY --chown=user:user "requirements.txt" "${HOME_DIR}/${APP_DIR}/requirements.txt"
RUN /bin/bash -c "source ${HOME_DIR}/tt-metal/build/python_env/bin/activate && pip install --default-timeout=240 --no-cache-dir -r requirements.txt" 

# run app via gunicorn
WORKDIR "${HOME_DIR}/${APP_DIR}/src"
# CMD sleep infinity
CMD ["/bin/bash", "-c", "source /home/user/tt-metal/build/python_env/bin/activate && gunicorn --config gunicorn.conf.py"]


# HEALTHCHECK --retries=5 --start-period=1s CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1