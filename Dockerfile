#FROM python:3.8
FROM pytorch/pytorch

WORKDIR /test_app

COPY  . /test_app

# RUN pip install --no-cache-dir --upgrade /code/requirements.txt

CMD ["python","test.py"]
