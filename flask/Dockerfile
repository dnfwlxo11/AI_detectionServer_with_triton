FROM python:latest
MAINTAINER Daechu

COPY . /app

WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg' 'libsm6' 'libxext6'  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
