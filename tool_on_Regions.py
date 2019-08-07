#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:51:00 2018

@author: gonthier
"""
import numpy as np
from tf_faster_rcnn.lib.model.nms_wrapper import nms

def reduce_to_k_regions(k,rois,roi_scores, fc7,new_nms_thresh,score_threshold,minimal_surface):
    """ Reduce the number of region to k or less 
    but it can even return more than k regions if we go out of the loop on new_nms_thresh
    """
    
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)
        
    keep = np.where(roi_scores> score_threshold)
    rois = rois[keep[0], :]
    roi_scores = roi_scores[keep]
    fc7 = fc7[keep[0],:]
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)

    width = rois[:,2] - rois[:,0] +1
    height = rois[:,3] - rois[:,1] +1
    surface = width*height
    keep = np.where(surface > minimal_surface)
    rois = rois[keep[0], :]
    roi_scores = roi_scores[keep]
    fc7 = fc7[keep[0],:]
    if(len(fc7) <= k):
        return(rois,roi_scores, fc7)
        
    #new_nms_thresh = 0.0
    keep_all = []
    for i in range(7):
        rois_plus_scores = np.hstack((rois[:,1:5],roi_scores.reshape(-1,1)))
        tmp_keep = nms(rois_plus_scores,new_nms_thresh)
        
        keep_new = np.setdiff1d(tmp_keep,keep_all) # Nouveau index
        
        keep_all2 = np.union1d(keep_all,tmp_keep) # sorted 
        if len(keep_all2) > k:
            keep = np.union1d(keep_all,keep_new[0:k-len(keep_all)]).astype(int)
            assert(len(keep)==k)
            rois = rois[keep, :]
            roi_scores = roi_scores[keep]
            fc7 = fc7[keep,:]
            assert(0 in keep)
            return(rois,roi_scores, fc7)
        else: 
            keep_all = keep_all2
            
        new_nms_thresh += 0.1

    return(rois,roi_scores, fc7)