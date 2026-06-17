# hadolint global ignore=DL3008,DL3013,DL3042
# FROM google/cloud-sdk:stable AS gcloud-source
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TRAINING_ENV=cloud
# Necessary flags for Nvidia CUDA on Google Cloud
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/:$LD_LIBRARY_PATH

# Google Cloud utils
# COPY --from=gcloud-source /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk
# ENV PATH="/usr/lib/google-cloud-sdk/bin:${PATH}"

# General utils
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    make \
    git \
    unzip \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*


# Sheetify
RUN mkdir /code
WORKDIR /code

COPY trainers/requirements_trainer_gpu.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY sheets /code/sheets
COPY Makefile /code/Makefile

# Mount maestro data folder to /code/data via Cloud Storage FUSE
CMD ["make", "run_train_basic"]
