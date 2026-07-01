# hadolint global ignore=DL3007,DL3008,DL3013,DL3042
FROM google/cloud-sdk:stable AS gcloud-source
FROM python:3.10.6-slim
ENV CLOUDSDK_PYTHON="/usr/bin/python3.10"

COPY --from=gcloud-source /usr/lib/google-cloud-sdk /usr/lib/google-cloud-sdk
ENV PATH="/usr/lib/google-cloud-sdk/bin:${PATH}"

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    make git unzip python3 \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir /code
WORKDIR /code

COPY trainers/requirements_trainer.txt /code/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
