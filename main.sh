#!/usr/bin/env bash
container_home=$1

#service crond start
python $container_home/osdemo/manage.py migrate --settings=OSdemo_django.settings
python $container_home/osdemo/credit_risk/credit_risk.py
gunicorn --config $container_home/gunicorn.conf.py --workers 1 OSdemo_django.wsgi:application