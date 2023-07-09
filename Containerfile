FROM python:3.9-slim-buster

ARG S3_ACCESS_KEY
ARG S3_SECRET_KEY

RUN echo "key: $S3_ACCESS_KEY"
WORKDIR /app
RUN mkdir -p /app
COPY ./requirements.txt /app
RUN ls /app
RUN apt update && \
apt install -y gcc && \
apt install -y git && \
apt-get install build-essential -y

RUN pip install --upgrade --no-cache-dir pip \
    && pip install --no-cache-dir -r  /app/requirements.txt \
    && apt-get purge -y  \
                -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/* \
        /tmp/*

COPY ./s3_clone.sh /app
COPY ./s3_models.txt /app
COPY ./get_model_from_s3.py /app
 RUN ls -lah /app
 RUN sh /app/s3_clone.sh
 RUN cd /app/s3_wrapper && pip3 install -e ./
 RUN python /app/get_model_from_s3.py
COPY ./ /app
 RUN ls -lah /app/
CMD ["uvicorn", "flask_server:app", "--reload", "--host", "0.0.0.0", "--port", "8989", "--backlog", "8", "--limit-concurrency", "8"]
