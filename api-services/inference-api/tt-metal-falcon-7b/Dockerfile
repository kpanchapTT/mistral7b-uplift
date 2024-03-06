FROM ubuntu:20.04

ENV SHELL=/bin/bash

USER root
WORKDIR /tmp/install

# ## set timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
RUN apt update && apt install -y \
    ## TT-Metalium Backend Dependencies: https://github.com/tenstorrent-metal/tt-metal/blob/main/INSTALLING.md#installing-dependencies-on-ubuntu-before-accelerator-level
    # RUN apt-get update && apt-get install -y \
    software-properties-common=0.99.9.12 \
    build-essential=12.8ubuntu1.1 \
    python3.8-venv=3.8.10-0ubuntu1~20.04.9 \
    libgoogle-glog-dev=0.4.0-1build1 \
    libyaml-cpp-dev=0.6.2-4ubuntu1 \
    libboost-all-dev=1.71.0.0ubuntu2 \
    libsndfile1=1.0.28-7ubuntu0.2 \
    libhwloc-dev \
    # other
    tzdata \
    # build-essential=12.8ubuntu1.1 \
    # clang-6.0 \
    # libboost-all-dev=1.71.0.0ubuntu2 \
    # libgoogle-glog-dev=0.4.0-1build1 \
    # libhdf5-serial-dev \
    # libsndfile1=1.0.28-7ubuntu0.2 \
    # libyaml-cpp-dev=0.6.2-4ubuntu1 \
    python-is-python3 \
    # pciutils \
    python3-pip \
    # software-properties-common \
    ## apt-get install user packages
    # RUN apt-get install -y \
    git \
    git-lfs \
    htop \
    nano \
    screen \
    tmux \
    vim \
    unzip \
    zip \
    ## apt-get install superpower packages
    # RUN apt-get install -y \
    curl \
    iputils-ping \
    rsync \
    wget \
    && rm -rf /var/lib/apt/lists/*

## pip install python requirements
# COPY requirements-metal.txt /tmp/install/
# RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
#     pip install -r /tmp/install/requirements-metal.txt && \
#     rm -rf /tmp/install


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
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    make build
    # pip install . --default-timeout=240 --no-cache-dir

## tt-smi
#COPY tt-smi_2023-06-16-0283a02404487eea /usr/local/bin/tt-smi
# COPY tt-smi-wh-8.D.0.0_2023-11-21-68fc880045868a27 /usr/local/bin/tt-smi
# COPY .bashrc .bash_aliases .inputrc .profile .vimrc /home/user/
# RUN chown -R user:user ${HOME_DIR}
# RUN chmod +x /usr/local/bin/tt-smi

WORKDIR "${HOME_DIR}/${APP_DIR}/src"
COPY "src" "${HOME_DIR}/${APP_DIR}/src"


CMD sleep infinity
