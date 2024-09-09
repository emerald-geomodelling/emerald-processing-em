#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:22:34 2022


"""

import numpy as np


def tempfilename(prefix='tmp'):
    number=str(np.random.randint(10000))
    filename="".join([prefix, number, ".tmp"])
    return filename


