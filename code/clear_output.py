#!/usr/bin/env python

import sys
import os
import subprocess


def strip_file(filepath):
    """Strip a notebook from its output inplace"""
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    if ext == '.ipynb':
        cmd = ('jupyter-nbconvert --inplace --to notebook '
               '--ClearOutputPreprocessor.enabled=True {} --output {}')
        subprocess.check_call(cmd.format(filepath, filename).split())

def strip_dir(path):
    """Strip notebooks in a directory and its subdirectories"""
    for dirpath, dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) != '.ipynb_checkpoints':
            if dirpath == path:
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    strip_file(filepath)

            for dirname in dirnames:
                strip_dir(os.path.join(dirpath, dirname))


if __name__ == '__main__':
    path = '.'
    if len(sys.argv) >= 2:
        path = sys.argv[1]

    if os.path.isfile(path):
        strip_file(path)
    else:
        strip_dir(path)
