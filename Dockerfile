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
RUN apt-get update && apt-get install libyaml-cpp0.6

ARG APP_DIR=/falcon40b-demo
WORKDIR "${APP_DIR}"
COPY "inference-api" "${APP_DIR}/inference-api"
COPY "requirements_minimal.txt" "${APP_DIR}"
COPY "run_inference_api.sh" "${APP_DIR}"

RUN pip3 install -r "requirements_minimal.txt"

# CMD ["bash", "run_inference_api.sh"]
ENTRYPOINT sleep infinity