FROM python:3.8.10

ENV CONTAINER_PREDICTION=/Prediction

WORKDIR $CONTAINER_PREDICTION

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . ./src

CMD [ "python3", "./src/app_prediction.py"]
