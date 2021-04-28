#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:30:11 2019

@author: gonthier
"""

import argparse
from TL_MIL import run_and_eval_MImax

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Weakly Supervised MI_Max model')
  parser.add_argument('--dataset', dest='dataset',
      help='training dataset',
      default='IconArt_v1', type=str) # IconArt_v1  comic watercolor CASPApaintings
  parser.add_argument('--net', dest='net',
      help='res152_COCO',
      default='res152_COCO', type=str)
  parser.add_argument('--path_data', dest='path_data',
      help='directory to data', default="data",
      type=str)
  parser.add_argument('--path_output', dest='path_output',
      help='directory to output', default="output",
      type=str)
  parser.add_argument('--path_to_model', dest='path_to_model',
      help='directory to pretrained model', default="model",
      type=str)
  parser.add_argument('--Optimizer', dest='Optimizer',
      help='directory to load models', default="GradientDescent",
      type=str)
  parser.add_argument('--loss_type', dest='loss_type',
      help='type of the loss : '' or hinge', default='',
      type=str)
  parser.add_argument('--CV_Mode', dest='CV_Mode',
      help='set to CV if one want to use the CV mode', default='',
      type=str)
  parser.add_argument('--k_per_bag', dest='k_per_bag',
      help='number of region considered per image',
      default=300, type=int)
  parser.add_argument('--LR', dest='LR',
      help='learning rate',
      default=0.01, type=float)
  parser.add_argument('--epsilon', dest='epsilon',
      help='epsilon when using score',
      default=0.01, type=float)
  parser.add_argument('--C', dest='C',
      help='regularization term',
      default=1.0, type=float)
  parser.add_argument('--mini_batch_size', dest='mini_batch_size',
      help='mini batch size if 0 then it will be 1000',
      default=0, type=int)
  parser.add_argument('--num_split', dest='num_split',
      help='number of split for the cross validation split',
      default=2, type=int)
  parser.add_argument('--restarts', dest='restarts',
      help='Number of reinitialization',
      default=11, type=int)
  parser.add_argument('--Polyhedral', dest='Polyhedral',
      help='Consider the polyhedral Mimax model.',
      action='store_true')
  parser.add_argument('--max_iters_all_base', dest='max_iters_all_base',
      help='Number of iterations',
      default=300, type=int)
  parser.add_argument('--PlotRegions', dest='PlotRegions',
      help='plot regions on the train and test images',
      action='store_true')
  parser.add_argument('--verbose', dest='verbose',
                      help='verbose mode',
                      action='store_true')
  parser.add_argument('--with_scores', dest='with_scores',
                      help='use the score',
                      action='store_true')
  parser.add_argument('--C_Searching', dest='C_Searching',
                      help='use different C values',
                      action='store_true')
  args = parser.parse_args()
  return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    # This function return the AP for the detection task with IoU > 0.5, IuO > 0.1 and for the classification task
    apsAt05,apsAt01,AP_classif = run_and_eval_MImax(demonet = args.net,database = args.dataset,
                                  ReDo = True,PlotRegions=args.PlotRegions,
                                  verbose = args.verbose,k_per_bag=args.k_per_bag,
                                  CV_Mode=args.CV_Mode,num_split=args.num_split,
                                  restarts=args.restarts,max_iters_all_base=args.max_iters_all_base,
                                  LR=args.LR,
                                  C=args.C,Optimizer=args.Optimizer,
                                  with_scores=args.with_scores,epsilon=args.epsilon,
                                  C_Searching=args.C_Searching,
                                  thresh_evaluation=0.05,TEST_NMS=0.3,
                                  mini_batch_size=args.mini_batch_size,loss_type=args.loss_type,
                                  path_data=args.path_data,path_output=args.path_output,
                                  path_to_model=args.path_to_model,Polyhedral=args.Polyhedral)
    
    
