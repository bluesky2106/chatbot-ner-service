FROM python:latest as build

RUN mkdir -p /usr/src/app/backend

WORKDIR /usr/src/app/backend

COPY . /usr/src/app/backend

RUN pip install -U -r requirements.txt

CMD python app.py