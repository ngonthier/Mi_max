#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:05:40 2018

@author: gonthier
"""

import numpy as np

def arrayToLatex(a,per=False,dtype=np.float32):
    if dtype==np.float32:
        if per:
            multi = 100.
        else:
            multi=1.
        stra = ' & '
        for i in range(len(a)):
            if per:
                stra += "{0:.1f} & ".format(a[i]*multi)
            else:
                stra += "{0:.3f} & ".format(a[i]*multi)
        if per:        
            stra += "{0:.1f} \\\ \hline".format(np.mean(a)*multi)
        else:
            stra += "{0:.3f} \\\ \hline".format(np.mean(a)*multi)
        return(stra)
    elif dtype==str:
        stra = ' & '
        for i in range(len(a)):
            stra +=a[i] +" & "
        stra += "mean \\\ \hline"
        return(stra)