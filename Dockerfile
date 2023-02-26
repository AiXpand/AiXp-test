#FROM python:3.8
FROM pytorch/pytorch

WORKDIR /test_app

COPY  . /test_app

RUN pip install --no-cache-dir --upgrade -r requirements.txt


ENV AIXP_DOCKER Yes

CMD ["python","test.py"]
