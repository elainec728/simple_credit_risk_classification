# Set the base image to python
FROM python:3.6
USER root
# Set variables for project name, and where to place files in container.
ENV CONTAINER_HOME=/credit \
    PROJECT=credit \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}:/credit" \
    PROJECT_ENV='dev' \
    DJANGO_SETTINGS_MODULE=osdemo.OSdemo_django.settings

# todo install jquery
EXPOSE 8000
RUN mkdir -p $CONTAINER_HOME \
    && apt-get update \
    && apt-get install -y git wget vim gcc

# Install Python dependencies
COPY requirements.txt $CONTAINER_HOME
RUN pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --upgrade pip \
    && pip install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com -r $CONTAINER_HOME/requirements.txt

ENV ENV_FOR_DYNACONF=dev
COPY credit $CONTAINER_HOME
WORKDIR $CONTAINER_HOME/osdemo
#CMD ['tail','-f','/etc/hosts']
CMD ["sh","-c","sh $CONTAINER_HOME/main.sh $CONTAINER_HOME"]