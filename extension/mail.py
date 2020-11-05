#!/usr/bin/env python3
# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import smtplib
from email.mime.text import MIMEText
from email.header import Header
# 附件
from email.mime.multipart import MIMEMultipart
# MIMEAuido MIMEImage MIMEText MIMEApplication
# from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.utils import parseaddr,formataddr

# 配置
from_addr='13776668123@163.com'
password='lhx990229'
smtp_server='smtp.163.com'
to_addr='8972167@qq.com'
port=25

#内容 plain
msg=MIMEText('send by python smtp','plain','utf-8')

#内容 html
msg=MIMEText('<html><body><h1>hello</h1>'+\
             '<p>send by <a href="http://www.python.org">python</a>...</p>'+\
             '</body></html>','html','utf-8')
# 添加 附件
# 邮件正文是MIMEText:
msg=MIMEMultipart()
msg.attach(MIMEText('send with file...', 'plain', 'utf-8'))
# 添加附件就是加上一个MIMEBase，从本地读取一个图片:
with open('/Users/michael/Downloads/c_test.png', 'rb') as f:
    # 设置附件的MIME和文件名，这里是png类型:
    mime = MIMEImage(f.read(), 'png', filename='c_test.png')
    # 加上必要的头信息:
    mime.add_header('Content-Disposition', 'attachment', filename='c_test.png')
    mime.add_header('Content-ID', '<0>')
    mime.add_header('X-Attachment-Id', '0')
    # 把附件的内容读进来:
    # mime.set_payload(f.read())
    # 用Base64编码:
    # encoders.encode_base64(mime)
    # 添加到MIMEMultipart:
    msg.attach(mime)
# 必须的
msg['Subject']=Header('smtp','utf-8')
msg['From']=from_addr
msg['To']=to_addr
# SSL port=994
server=smtplib.SMTP_SSL(smtp_server, 994)
server.set_debuglevel(2)
# identify yourself to esmtp using ehlo
server.ehlo()

# TLS  transport layer security 25
server=smtplib.SMTP(smtp_server,port)
server.set_debuglevel(2)
server.ehlo()
server.starttls()
server.ehlo()

# 明文
server=smtplib.SMTP(smtp_server,port)
server.set_debuglevel(2)
# identify yourself to esmtp using ehlo
server.ehlo()

server.login(from_addr,password)
server.sendmail(from_addr,to_addr,msg.as_string())
server.quit()
