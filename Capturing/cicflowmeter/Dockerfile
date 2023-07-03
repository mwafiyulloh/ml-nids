FROM python:3.8.10

ENV CONTAINER_CAPTURING=/Capturing

WORKDIR $CONTAINER_CAPTURING

RUN apt-get update && apt-get install -y tcpdump

ADD . ${CONTAINER_CAPTURING}
RUN python3 setup.py install

CMD ["python3", "start_generate_tmpfile.py"]
