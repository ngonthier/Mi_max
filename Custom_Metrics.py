#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:59:33 2017

@author: gonthier
"""

import numpy as np

def VOCevalaction(y_true, y_score):

    si = np.argsort(y_score)[::-1]
    tp=y_true[si]>0
    fp=y_true[si]<=0
    
    fp= np.cumsum(fp)
    tp= np.cumsum(tp)
    
    #print(fp,tp)
    
    rec= tp/np.sum(y_true>0)
    prec= tp/(fp+tp)
    
    #print(rec,prec)
    
    return(rec,prec)
    

def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    From https://gist.github.com/mblondel/7337391
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)

def computeAveragePrecision(recalls, precisions, use_07_metric=False):
    """ ap = voc_ap(recalls, precisions, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    From https://github.com/Azure/ObjectDetectionUsingCntk/blob/34c73c254dd6925a8c09911415d8427d4f02a8e0/helpers.py
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrecalls = np.concatenate(([0.], recalls, [1.]))
        mprecisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(mprecisions.size - 1, 0, -1):
            mprecisions[i - 1] = np.maximum(mprecisions[i - 1], mprecisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrecalls[1:] != mrecalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrecalls[i + 1] - mrecalls[i]) * mprecisions[i + 1])
    return ap