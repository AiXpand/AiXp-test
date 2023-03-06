#FROM python:3.8
FROM aixpand/exe_eng_pub:x64_env_full

WORKDIR /test_app

COPY  . /test_app

ENV TZ Europe/Bucharest

ENV AIXP_DOCKER Yes
ENV EE_ID E2DkrTester
ENV SHOW_PACKS Yes

CMD ["python","test.py"]
