# Code from pybuda release pipeline: https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/blob/cdbd30a6ddb4ac9d2c27450dae5ad9c129018e9d/ci/deployment
#############################################
FROM ubuntu:20.04 AS pybuda-base

RUN apt-get update

#############################################
##  Buda Backend Dependencies
#############################################
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:rosecompiler/rose-development
RUN apt-get install -y rose rose-tools ruby libgoogle-glog-dev python3.8-venv
RUN apt-get install -y build-essential clang-6.0 libhdf5-serial-dev
RUN apt-get install -y python3-pip libzmq3-dev
RUN apt-get install -y python-is-python3
RUN apt-get install -y pciutils

RUN pip3 install pyzmq tabulate

#############################################
##  This image depends on presence of 
##  1. pybuda and tvm wheel files
#############################################
ARG PYBUDA_WHEEL=pybuda-0.1.231113+dev.wh.b0.cdbd30a-cp38-cp38-linux_x86_64.whl
ARG TVM_WHEEL=tvm-0.9.0+dev.tt.c2076affc-cp38-cp38-linux_x86_64.whl

RUN test -n "$PYBUDA_WHEEL" || (echo "PYBUDA_WHEEL variable not set" && false)
RUN test -n "$TVM_WHEEL" || (echo "TVM_WHEEL variable not set" && false)

COPY  ${PYBUDA_WHEEL} .
COPY  ${TVM_WHEEL} .

#############################################
## Install PyBuda and TVM
#############################################
RUN pip3 install ${PYBUDA_WHEEL} --default-timeout=120
RUN pip3 install ${TVM_WHEEL} --default-timeout=120

#############################################
# project-falcon specific
#############################################
# additonal dependencies:
# libyaml-cpp0.6: for pybuda
# rsync: to save .tti
RUN apt-get update && apt-get install -y libyaml-cpp0.6 rsync

ARG APP_DIR=/falcon40b-demo
ARG HOME_DIR=/home/user

## setup user
RUN useradd -u 1000 -s /bin/bash user

# add user to sudoers
RUN apt-get install -y sudo \
    && echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user

WORKDIR "${HOME_DIR}/${APP_DIR}"

COPY "inference-api" "${HOME_DIR}/${APP_DIR}/inference-api"
COPY "requirements.txt" "${HOME_DIR}/${APP_DIR}/"
COPY "run_inference_api.sh" "${HOME_DIR}/${APP_DIR}/"

RUN chown -R user:user "${HOME_DIR}"
# BBE writes files here
RUN chown -R user:user "/usr/local/lib/python3.8/dist-packages/budabackend"

# debug tools
RUN apt-get update && apt-get install -y vim tmux curl htop zip unzip
COPY "tt-smi-wh-8.C.0.0_2023-11-02-ddcfb4b7bb67635e" "${HOME_DIR}/"
COPY "soft_harvest_2023-09-14-3d569476654ec0cd" "${HOME_DIR}/"

RUN chmod +x "${HOME_DIR}/tt-smi-wh-8.C.0.0_2023-11-02-ddcfb4b7bb67635e"
RUN chmod +x "${HOME_DIR}/soft_harvest_2023-09-14-3d569476654ec0cd"

RUN pip3 install -r "requirements.txt" --default-timeout=120

# cleanup
RUN rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/install

USER user
# CMD ["bash", "run_inference_api.sh"]
ENTRYPOINT sleep infinity