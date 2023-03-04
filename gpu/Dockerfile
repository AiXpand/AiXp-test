#FROM python:3.8
FROM aixpand/exe_eng_pub:env_ext

WORKDIR /gpu_test

COPY  . /gpu_test

ENV AIXP_DOCKER Yes
ENV EE_ID E2DkrTester
ENV SHOW_PACKS Yes

CMD ["python","main.py"]
