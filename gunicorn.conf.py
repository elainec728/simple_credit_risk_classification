#!/usr/bin/python
# -*- coding: utf-8 -*-
# =====================================
# @Author: elaine.cao@hexasino.com
# @Date: 11/22/2018 11:15 AM
# =====================================
bind = '0.0.0.0:8000'
# backlog = 512                #监听队列 等待服务的客户的数量
container_home = '/credit'  # TODO pass in
chdir = container_home + '/osdemo'
timeout = 200
worker_class = 'gevent'
preload = True

workers = 1
threads = 2
loglevel = 'info'
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
log_file = '-'
# accesslog = container_home + "/lcims/resources/log/gunicorn_access.log"  # 访问日志文件
# errorlog = container_home + "/lcims/resources/log/gunicorn_error.log"  # 错误日志文件