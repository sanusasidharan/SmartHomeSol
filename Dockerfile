FROM python:3.7

USER root

RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 -y --no-install-recommends \
    -y libxrender-dev \
    -y mesa-common-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

#WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

#permission issue
USER root
RUN chmod 777 -R /public

EXPOSE 8000

CMD ["python", "/server.py"]