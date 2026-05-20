# File name : basic_tools.py
# Author : Simon Bastide
# Encoding : utf-8
# Function : Common basic tools 
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
from datetime import datetime

def get_date_iso():
    return (datetime.now().replace().isoformat()).split('T')[0]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def match(s1, s2):
    # Check if two string match allowing one character difference
    # https://stackoverflow.com/questions/25216328/compare-strings-allowing-one-character-difference
    # Solution from Marco Sulla
    ok = False

    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            if ok:
                return False
            else:
                ok = True

    return ok


