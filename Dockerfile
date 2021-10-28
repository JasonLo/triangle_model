FROM tensorflow/tensorflow:nightly-gpu
ENV PYTHONUNBUFFERED True
ENV TF_ROOT /triangle_model
ENV PYTHONPATH /triangle_model/src
ENV GOOGLE_APPLICATION_CREDENTIALS /triangle_model/secret/gcp.json
WORKDIR /triangle_model
COPY requirements.txt ./
RUN pip install -r requirements.txt
