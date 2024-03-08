FROM ubuntu:20.04

ENV SHELL=/bin/bash

USER root
WORKDIR /tmp/install

# ## set timezone
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

## clone tt-metal and its submodules
ARG APP_DIR=/tt-metal-falcon-7b
ARG HOME_DIR=/home/user
WORKDIR ${HOME_DIR}

## add user
RUN useradd -u 1000 -s /bin/bash user
RUN chown -R user:user ${HOME_DIR}

USER user

ENV PATH=$PATH:${HOME_DIR}/.local/bin
ENV ARCH_NAME=grayskull
# ENV ARCH_NAME=wormhole_b0
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV TT_METAL_ENV=dev
ENV PYTHONPATH=${HOME_DIR}/tt-metal
# NOTE: production build is not documented, unclear if functional
# ENV TT_METAL_ENV=production
# ENV TT_METAL_CREATE_STATIC_LIB=1

RUN git clone https://github.com/tenstorrent-metal/tt-metal.git \
    --depth 1 \
    --recurse-submodules \
    --shallow-submodules \
    --branch v0.44.0 && \
    cd tt-metal && \
    git submodule foreach 'git lfs fetch --all && git lfs pull'

## build tt-metal
WORKDIR ${HOME_DIR}/tt-metal
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
RUN make build

WORKDIR "${HOME_DIR}/${APP_DIR}/src"
COPY "src" "${HOME_DIR}/${APP_DIR}/src"

CMD ["gunicorn", "--config", "gunicorn.conf.py"]

# default port is 7000
ENV SERVICE_PORT=7000 
HEALTHCHECK --retries=5 --start-period=1s CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
