# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import shutil, glob,os, sys


# def glob_files(_tdx_dir, file_name):
#     # e.g.  r'D:\通达信-1m\*'
#     print('_tdx_dir', _tdx_dir)
#     sh_path = os.path.join(_tdx_dir, 'vipdoc\sh\{}'.format(file_name))
#     print('sh_path', sh_path)
#     sz_path = os.path.join(_tdx_dir, 'vipdoc\sz\{}'.format(file_name))
#     print('sz_path', sz_path)
#     sh_files = glob.glob(sh_path)
#     sz_files = glob.glob(sz_path)
#     tdx_file_paths = sh_files + sz_files
#     print('tdx_file_paths', len(tdx_file_paths), tdx_file_paths[:10])
#     for file_path in tdx_file_paths:
#         shutil.rmtree(file_path)


# def glob_files(_tdx_dir, file_name):
#     # e.g.  r'D:\通达信-1m\*'
#     path = os.path.join(_tdx_dir, 'vipdoc\{}'.format(file_name))
#     print('path', path)
#     files = glob.glob(path)
#     print('files', len(files), files)
#     for file_path in files:
#         shutil.rmtree(file_path)
#     return files


# def on_error(f, path, exeinfo):
#     f(path)


def glob_files(_tdx_dir):
    # e.g.  r'D:\通达信-1m\*'
    print('_tdx_dir', _tdx_dir)
    sh_path = os.path.join(_tdx_dir, 'vipdoc\sh\minline\sh3*')
    sh_files = glob.glob(sh_path)
    print('sh_files', sh_files[:10])
    sz_path = os.path.join(_tdx_dir, 'vipdoc\sz\minline\sz2*')
    sz_files = glob.glob(sz_path)
    print('sz_files', sz_files[:10])
    tdx_file_paths = sh_files + sz_files
    print('tdx_file_paths', len(tdx_file_paths), tdx_file_paths[:10])
    for file_path in tdx_file_paths:
        try:
            # shutil.rmtree(file_path, onerror=on_error(os.remove, file_path, sys.exc_info))
            os.remove(file_path)
        except FileNotFoundError:
            pass


if __name__ == '__main__':

    tdx = r'D:\通达信-1m\*'
    # glob_files(tdx, 'tick')
    glob_files(tdx)

