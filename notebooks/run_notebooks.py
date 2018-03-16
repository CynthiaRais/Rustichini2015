#!/usr/bin/env python

import sys
import os
import time
import subprocess


this_dir = os.path.abspath(os.path.dirname(__file__))

def execute(filepath, timeout=10000):
    prefix, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    os.makedirs('html/', exist_ok=True)
    if ext == '.ipynb':
        # .html file
        cmd = ('jupyter-nbconvert', '--ExecutePreprocessor.timeout={}'.format(timeout),
               '--execute', '--to', 'html', filepath,
               '--output', '{}.html'.format(os.path.join(prefix, 'html', name)))
        print(' '.join(cmd))
        subprocess.check_call(cmd)


def execute_folder(path):
    for dirpath, dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) != '.ipynb_checkpoints':
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                execute(filepath)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        filepaths = [this_dir]
        filepaths = ['Figure_4.ipynb', 'Figure_5.ipynb', 'Figure_6.ipynb',
                     'Figure_7.ipynb']
    else:
        filepaths = sys.argv[1:]

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print("error: {} not found".format(filepath))
            sys.exit(1)

    for filepath in filepaths:
        if os.path.isfile(filepath):
            execute(filepath)
        else:
            execute_folder(filepath)
