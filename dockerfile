FROM tensorflow/tensorflow:2.14.0-gpu

RUN apt-get update
RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements-slim.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/app"
