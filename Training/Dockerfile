FROM python:3.8.10

ENV CONTAINER_TRAINING=/Training

WORKDIR $CONTAINER_TRAINING

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ./src

CMD [ "python3", "./src/app_training.py"]
