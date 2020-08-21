# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
def init_logging():
    """
    logging相关初始化工作，配置log级别，默认写入路径，输出格式
    """
    if g_is_ipython and not g_is_py3:
        """ipython在python2的一些版本需要reload logging模块，否则不显示log信息"""
        # noinspection PyUnresolvedReferences, PyCompatibility
        reload(logging)
        # pass

    if not os.path.exists(g_project_log_dir):
        # 创建log文件夹
        os.makedirs(g_project_log_dir)

    # 输出格式规范
    # file_handler = logging.FileHandler(g_project_log_info, 'a', 'utf-8')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=g_project_log_info,
                        filemode='a'
                        # handlers=[file_handler]
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 屏幕打印只显示message
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)