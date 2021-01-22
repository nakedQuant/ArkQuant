#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
"""
    Werkzeug 用于实现 WSGI ，应用和服务之间的标准 Python 接口。
    Jinja 用于渲染页面的模板语言。
    MarkupSafe 与 Jinja 共用，在渲染页面时用于避免不可信的输入，防止注入攻击。
    ItsDangerous 保证数据完整性的安全标志数据，用于保护 Flask 的 session cookie.
    Click 是一个命令行应用的框架。用于提供 flask 命令，并允许添加自定义 管理命令。
    
    FLASK_APP  FLASK_ENV 
    
    string （缺省值） 接受任何不包含斜杠的文本
    int 接受正整数
    float  接受正浮点数
    path 类似 string ，但可以包含斜杠
    uuid  接受 UUID 字符串
    
    http：
    1GET将数据以未加密的形式发送到服务器，这最常用的方法。
    2HEAD与GET相同，但没有响应主体
    3POST用于将HTML表单数据发送到服务器。通过POST方法接收的数据不会被服务器缓存。
    4PUT用上传的内容替换目标资源的所有当前表示。
    5DELETE删除由URL给出的所有目标资源的所有表示
    默认情况下，Flask路由响应GET请求。 但是，可以通过为route()装饰器提供方法参数来更改此首选项。
    为了演示在URL路由中使用POST方法，首先创建一个HTML表单并使用POST方法将表单数据发送到URL。
    
"""
# 实例化
import os
from flask import Flask, escape, url_for, request, render_template, redirect, flash, send_from_directory, make_response, abort
from werkzeug.utils import secure_filename


app = Flask(__name__)


@app.route('/')
def hello_world():
    print('hello world')
    return 'hello world debug'


@app.route('/validate')
def validate():
    return 'test_validate'


@app.route('/test_url')
def test_url():
    return 'test_url_connection'


@app.route('/user/<username>')
def show_url_name(username):
    return 'show_name_{}'.format(username)


@app.route('/post/<int:idx>')
def post_id(idx):
    return 'post_id:{}'.format(idx)


@app.route('/spath/<path:subpath>')
def show_path(subpath):
    return 'show_path_%s' % escape(subpath)


# post 请求
@app.route('/login', methods=['post'])
def login():
    uname = request.form['uname']
    password = request.form['pass']
    if uname == 'liuhx' and password == 'arkQuant':
        return 'test post'


@app.route('/_login', methods=['GET', 'POST'])
def _login():
    if request.method == 'POST':
        return validate()
    else:
        return test_url()


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


# 实现file upload
upload_folder = '/Users/python/PycharmProjects/ArkQuant/web'
allowed_extension = ['csv', 'excel', 'pdf', 'png', 'gif']

app.config['upload_folder'] = upload_folder
# 设定文件大小 ；文件很小的存入内存， 否则tempfile.gettempdir()
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extension


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['upload_folder'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
    '''


# 展示在网页
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['upload_folder'],
                               filename)

# # 读取cookies
# @app.route('/req_cookie')
# def index():
#     # req = request.cookies.get('username', 'default')
#     req = request.cookies
#     return req

# # error
# @app.route('/index')
# def index():
#     resp = make_response(render_template(...))
#     resp.set_cookie('username', 'liu heng xin')
#     return resp


@app.route('/index')
def index():
    return redirect(url_for('index_login'))


@app.route('/index_login')
def index_login():
    abort(401)
    # this_is_never_executed()