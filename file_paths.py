#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 01:08:01 2019

@author: Sruti
"""

import os

cwd =  os.getcwd()
filenames = os.listdir("data")
pathname = os.path.join("data", filenames[0])
open(pathname)
open file 'data/people.txt', mode 'r' at 0x7f74ad5d8270>