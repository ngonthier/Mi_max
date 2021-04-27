#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:45:35 2018

Script pour realiser du transfert d'apprentissage a partir de Faster RCNN

Il faut rajouter l elaboration de probabilite :
    https://stats.stackexchange.com/questions/55072/svm-confidence-according-to-distance-from-hyperline

Page utile sur VOC 2007 :
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html

@author: gonthier
"""

import time

import pickle
import tensorflow as tf
from tf_faster_rcnn.lib.model.test import get_blobs
from tf_faster_rcnn.lib.model.nms_wrapper import nms
#from tf_faster_rcnn.lib.nms.py_cpu_nms import py_cpu_nms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import average_precision_score,recall_score,precision_score,f1_score
from Custom_Metrics import ranking_precision_score
import os.path
from Mimax_model import tf_MI_max 
from LatexOuput import arrayToLatex
from FasterRCNN import vis_detections_list,Compute_Faster_RCNN_features
import pathlib
from tf_faster_rcnn.lib.datasets.factory import get_imdb
from IMDB import get_database

CLASSESVOC = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSESCOCO = ('__background__','person', 'bicycle','car','motorcycle', 'aeroplane','bus','train','truck','boat',
 'traffic light','fire hydrant', 'stop sign', 'parking meter','bench','bird',
 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
 'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball', 'kite',
 'baseball bat','baseball glove','skateboard', 'surfboard','tennis racket','bottle', 
 'wine glass','cup','fork', 'knife','spoon','bowl', 'banana', 'apple','sandwich', 'orange', 
'broccoli','carrot','hot dog','pizza','donut','cake','chair', 'couch','potted plant','bed',
 'diningtable','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
 'oven','toaster','sink','refrigerator', 'book','clock','vase','scissors','teddy bear',
 'hair drier','toothbrush')


NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',)
    ,'vgg16_coco': ('/media/gonthier/HDD/models/tf-faster-rcnn/vgg16/vgg16_faster_rcnn_iter_1190000.ckpt',)    
    ,'res101': ('res101_faster_rcnn_iter_110000.ckpt',)
    ,'res152' : ('res152_faster_rcnn_iter_1190000.ckpt',)}

DATASETS= {'coco': ('coco_2014_train+coco_2014_valminusminival',),'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

NETS_Pretrained = {'vgg16_VOC07' :'vgg16_faster_rcnn_iter_70000.ckpt',
                   'vgg16_VOC12' :'vgg16_faster_rcnn_iter_110000.ckpt',
                   'vgg16_COCO' :'vgg16_faster_rcnn_iter_1190000.ckpt',
                   'res101_VOC07' :'res101_faster_rcnn_iter_70000.ckpt',
                   'res101_VOC12' :'res101_faster_rcnn_iter_110000.ckpt',
                   'res101_COCO' :'res101_faster_rcnn_iter_1190000.ckpt',
                   'res152_COCO' :'res152_faster_rcnn_iter_1190000.ckpt'
                   }
CLASSES_SET ={'VOC' : CLASSESVOC,
              'COCO' : CLASSESCOCO }

depicts_depictsLabel = {'Q942467_verif': 'Jesus_Child','Q235113_verif':'angel_Cupidon ','Q345_verif' :'Mary','Q109607_verif':'ruins','Q10791_verif': 'nudity'}

def parser_w_mei_reduce(record,num_rois=300,num_features=2048):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'score_mei': tf.FixedLenFeature([1], tf.float32),
                'mei': tf.FixedLenFeature([1], tf.int64),
                'rois': tf.FixedLenFeature([num_rois*5],tf.float32),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'fc7_selected': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([1],tf.float32),
                'name_img' : tf.FixedLenFeature([],tf.string)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
    label_300 = tf.tile(label,[num_rois])
    fc7_selected = parsed['fc7_selected']
    fc7_selected = tf.reshape(fc7_selected, [num_rois,num_features])         
    return fc7_selected,label_300

def parser_w_rois(record,classe_index=0,num_classes=10,num_rois=300,num_features=2048,
                  dim_rois=5):
    # Perform additional preprocessing on the parsed data.
    keys_to_features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'num_regions':  tf.FixedLenFeature([], tf.int64),
                'num_features':  tf.FixedLenFeature([], tf.int64),
                'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                'label' : tf.FixedLenFeature([num_classes],tf.float32),
                'name_img' : tf.FixedLenFeature([],tf.string)}
    parsed = tf.parse_single_example(record, keys_to_features)
    
    # Cast label data into int32
    label = parsed['label']
    name_img = parsed['name_img']
    label = tf.slice(label,[classe_index],[1])
    label = tf.squeeze(label) # To get a vector one dimension
    fc7 = parsed['fc7']
    fc7 = tf.reshape(fc7, [num_rois,num_features])
    rois = parsed['rois']
    rois = tf.reshape(rois, [num_rois,dim_rois])           
    return fc7,rois, label,name_img

def parser_w_rois_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            with_rois_scores=False,dim_rois=5):
        # Perform additional preprocessing on the parsed data.
        if not(with_rois_scores):
            keys_to_features={
                        'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}
        else:
            keys_to_features={
                        'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                        'rois': tf.FixedLenFeature([num_rois*dim_rois],tf.float32),
                        'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                        'label' : tf.FixedLenFeature([num_classes],tf.float32),
                        'name_img' : tf.FixedLenFeature([],tf.string)}
#        keys_to_features={
#                    'height': tf.FixedLenFeature([], tf.int64),
#                    'width': tf.FixedLenFeature([], tf.int64),
#                    'num_regions':  tf.FixedLenFeature([], tf.int64),
#                    'num_features':  tf.FixedLenFeature([], tf.int64),
#                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
#                    'rois': tf.FixedLenFeature([5*num_rois],tf.float32),
#                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
#                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
#                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
#                    'name_img' : tf.FixedLenFeature([],tf.string)}
            
        parsed = tf.parse_single_example(record, keys_to_features)
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [num_rois,num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [num_rois,dim_rois])    
        if not(with_rois_scores):
            return fc7,rois, label,name_img
        else:
            roi_scores = parsed['roi_scores'] 
            return fc7,rois,roi_scores,label,name_img
        
def parser_all_elt_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            dim_rois=5,noReshape=True):
    keys_to_features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'dim1_rois':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([dim_rois*num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
            
    parsed = tf.parse_single_example(record, keys_to_features)
      # Cast label data into int32
    list_elt = []
    for key in keys_to_features.keys():
          list_elt += [parsed[key]]
    if not(noReshape):
            list_elt[7] = tf.reshape(list_elt[7], [num_rois,num_features])
            list_elt[5] = tf.reshape(list_elt[5], [num_rois,dim_rois])  
        
    return(list_elt)
    
def parser_minimal_elt_all_class(record,num_classes=10,num_rois=300,num_features=2048,
                            dim_rois=5,noReshape=True):
    keys_to_features={
                    'rois': tf.FixedLenFeature([dim_rois*num_rois],tf.float32),
                    'roi_scores':tf.FixedLenFeature([num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([num_rois*num_features],tf.float32),
                    'label' : tf.FixedLenFeature([num_classes],tf.float32),
                    'name_img' : tf.FixedLenFeature([],tf.string)}
            
    parsed = tf.parse_single_example(record, keys_to_features)
      # Cast label data into int32
    list_elt = []
    for key in keys_to_features.keys():
          list_elt += [parsed[key]]
    if not(noReshape):
            list_elt[7] = tf.reshape(list_elt[2], [num_rois,num_features])
            list_elt[5] = tf.reshape(list_elt[0], [num_rois,dim_rois])  
        
    return(list_elt)


def run_and_eval_MImax(demonet = 'res152_COCO',database = 'IconArt_v1', ReDo = True,PlotRegions=False,
                                  verbose = True,k_per_bag=300,
                                  CV_Mode=None,num_split=2,
                                  restarts=11,max_iters_all_base=300,LR=0.01,
                                  C=1.0,Optimizer='GradientDescent',
                                  with_scores=False,epsilon=0.0,
                                  C_Searching=False,
                                  thresh_evaluation=0.05,TEST_NMS=0.3,
                                  mini_batch_size=None,loss_type='',
                                  path_data='data',path_output='output',path_to_model='models',
                                  Polyhedral=False):
    """ 
    This function used TFrecords file 
    
    Classifier based on CNN features with Transfer Learning on Faster RCNN output
    for weakly supervised object detection
    
    Note : with a features maps of 2048, k_bag =300 and a batchsize of 1000 we can 
    train up to 1200 W vectors in parallel at the same time on a NVIDIA 1080 Ti
    
    @param : demonet : the kind of inside network used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : database : the database used for the weakly supervised detection task
    @param : verbose : Verbose option classical
    @param : ReDo = False : Erase the former computation, if True and the model already exists
        : only do the evaluation
    @param : PlotRegions : plot the regions used for learn and the regions in 
        the positive output response
    @param : k_per_bag : number of element per batch in the slection phase [defaut : 300] 
    @param : CV_Mode : cross validation mode in the MI_max : possibility ; 
        None, CV in k split 
    @param : num_split  : Number of split for the CV 
    @param : restarts  :  number of restarts / reinitialisation in the MI_max [default=11]
    @param : max_iters_all_base  :  number of maximum iteration on the going on 
        the full database 
    @param : LR  :  Learning rate for the optimizer in the MI_max 
    @param : C  :  Regularisation term for the optimizer in the MI_max 
    @param : Optimizer  : Optimizer for the MI_max GradientDescent or Adam
    @param : thresh_evaluation : 0.05 : seuillage avant de fournir les boites a l evaluation de detections
    @param : TEST_NMS : 0.3 : recouvrement autorise avant le NMS avant l evaluation de detections
    @param : mini_batch_size if None or 0 an automatic adhoc mini batch size is set
    @param : Polyhedral consider the polyhedral model 
    
    This function output AP for different dataset for the weakly supervised task 
    
    """
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label = get_database(database) 
    num_trainval_im = len(df_label[df_label['set']=='train'][item_name]) + len(df_label[df_label['set']==str_val][item_name])
   
    if verbose: print('Training on ',database,'with ',num_trainval_im,' images in the trainval set')
    N = 1
    extL2 = ''
    nms_thresh = 0.7
    savedstr = '_all'
    metamodel='FasterRCNN'

    sets = ['trainval','test']
    dict_name_file = {}
    data_precomputeed= True
    if k_per_bag==300:
        k_per_bag_str = ''
    else:
        k_per_bag_str = '_k'+str(k_per_bag)
    for set_str in sets:
        name_pkl_all_features =  os.path.join(path_output,metamodel+'_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords')
        dict_name_file[set_str] = name_pkl_all_features
        if not(os.path.isfile(name_pkl_all_features)):
            data_precomputeed = False

    if demonet in ['vgg16_COCO','vgg16_VOC07','vgg16_VOC12']:
        num_features = 4096
    elif demonet in ['res101_COCO','res152_COCO','res101_VOC07','res152']:
        num_features = 2048
    
    if not(data_precomputeed):
        # Compute the features
        if verbose: print("We will use a Faster RCNN as feature extractor and region proposals")
        if metamodel=='FasterRCNN':
            Compute_Faster_RCNN_features(demonet=demonet,nms_thresh =nms_thresh,
                                         database=database,verbose=verbose,
                                         k_regions=k_per_bag,path_data=path_data,
                                         path_output=path_output,
                                         path_to_model=path_to_model)
        else:
            raise(NotImplementedError)
 
    # Config param for TF session 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  
            
    # Data for the MI_max Latent SVM
    # All those parameter are design for a GPU 1080 Ti memory size  ie 11GB
    performance = False

    sizeMax = 30*200000 // (k_per_bag*20)
    if not(CV_Mode=='CV' and num_split==2):
        sizeMax //= 2
    if num_features > 2048:
        sizeMax //= (num_features//2048)
    
    model_str = 'MI_max'
    if k_per_bag==300:
        buffer_size = 10000
    else:
        buffer_size = 5000*300 // k_per_bag

    if (k_per_bag > 300 or num_trainval_im > 5000):
        usecache = False
    else:
        usecache = True

    if mini_batch_size is None or mini_batch_size==0:
        mini_batch_size = min(sizeMax,num_trainval_im)

    max_iters = ((num_trainval_im // mini_batch_size)+ \
                 np.sign(num_trainval_im % mini_batch_size))*max_iters_all_base
            
    AP_per_class = []
    P_per_class = []
    R_per_class = []
    P20_per_class = []
    AP_per_classbS = []
    final_clf = None
    
    if C == 1.0:
        C_str=''
    else:
        C_str = '_C'+str(C) # regularisation term 
    if with_scores:
        with_scores_str = '_WRC'+str(epsilon)
    else:
        with_scores_str=''
    
    extPar = '_p'
    if CV_Mode=='CV':
        max_iters = (max_iters*(num_split-1)//num_split) # Modification d iteration max par rapport au nombre de split
        extCV = '_cv'+str(num_split)
    elif CV_Mode is None or CV_Mode=='':
        extCV =''
    else:
        raise(NotImplementedError)
    extCV += '_wr'

    if Optimizer=='Adam':
        opti_str=''
    elif Optimizer=='GradientDescent':
        opti_str='_gd'
    elif Optimizer=='lbfgs':
        opti_str='_lbfgs'
    else:
        raise(NotImplementedError)
    
    if loss_type is None or loss_type=='':
        loss_type_str =''
    elif loss_type=='hinge':
        loss_type_str = 'Losshinge'
    
    if LR==0.01:
        LR_str = ''
    else:
        LR_str='_LR'+str(LR)
    
    optimArg = None
    if optimArg== None or Optimizer=='GradientDescent':
        optimArg_str = ''
    else:
        if  Optimizer=='Adam' and str(optimArg).replace(' ','_')=="{'learning_rate':_0.01,_'beta1':_0.9,_'beta2':_0.999,_'epsilon':_1e-08}":
            optimArg_str = ''
        else:
            optimArg_str =  str(optimArg).replace(' ','_')
    verboseMI_max = verbose
    shuffle = True
    if num_trainval_im==mini_batch_size:
        shuffle = False

    number_zone = k_per_bag

    dont_use_07_metric = True

    dim_rois = 5
  
    cachefilefolder = os.path.join(path_output,'cachefile')

    if Polyhedral: 
        poly_str = '_Poly'
    else:
        poly_str = ''

    cachefile_model_base='WLS_'+ database+ '_'+demonet+'_r'+str(restarts)+'_s' \
        +str(mini_batch_size)+'_k'+str(k_per_bag)+'_m'+str(max_iters)+extPar+\
        extCV+opti_str+LR_str+C_str+with_scores_str+ loss_type_str+poly_str
    pathlib.Path(cachefilefolder).mkdir(parents=True, exist_ok=True)
    cachefile_model = os.path.join(cachefilefolder,cachefile_model_base+'_'+model_str+'.pkl')

    if verbose: print("cachefile name",cachefile_model)
    if not os.path.isfile(cachefile_model) or ReDo:
        name_milsvm = {}
        if verbose: print("The cachefile doesn t exist or we will erase it.")    
    else:
        with open(cachefile_model, 'rb') as f:
            name_milsvm = pickle.load(f)
            if verbose: print("The cachefile exists")
    
    usecache_eval = True

    if database=='watercolor':
        imdb = get_imdb('watercolor_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index)
    elif database=='PeopleArt':
        imdb = get_imdb('PeopleArt_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index)
    elif database=='clipart':
        imdb = get_imdb('clipart_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images = len(imdb.image_index) 
    elif database=='IconArt_v1':
        imdb = get_imdb('IconArt_v1_test')
        imdb.set_force_dont_use_07_metric(dont_use_07_metric)
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    
    else:
        num_images =  len(df_label[df_label['set']=='test'][item_name])
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
   
    data_path_train= dict_name_file['trainval']
    
    if not os.path.isfile(cachefile_model) or ReDo:
         if verbose: t0 = time.time()

         
         classifierMI_max = tf_MI_max(LR=LR,C=C,restarts=restarts,num_rois=k_per_bag,
               max_iters=max_iters,buffer_size=buffer_size,
               verbose=verboseMI_max,Optimizer=Optimizer,
               mini_batch_size=mini_batch_size,num_features=num_features,
               num_classes=num_classes,num_split=num_split,CV_Mode=CV_Mode,with_scores=with_scores,
               epsilon=epsilon,loss_type=loss_type,usecache=usecache,Polyhedral=Polyhedral)
         export_dir = classifierMI_max.fit_MI_max_tfrecords(data_path=data_path_train, \
               shuffle=shuffle,C_Searching=C_Searching)  
             
         if verbose: 
             t1 = time.time() 
             print('Total duration training part :',str(t1-t0))
              
         np_pos_value,np_neg_value = classifierMI_max.get_porportions()
         name_milsvm =export_dir,np_pos_value,np_neg_value
         with open(cachefile_model, 'wb') as f:
             pickle.dump(name_milsvm, f)
    else:
        if verbose: print("We will load the existing model")
        export_dir,np_pos_value,np_neg_value= name_milsvm   

    true_label_all_test,predict_label_all_test,name_all_test,labels_test_predited \
    ,all_boxes = \
    tfR_evaluation_parall(database=database,num_classes=num_classes,
               export_dir=export_dir,dict_name_file=dict_name_file,mini_batch_size=mini_batch_size
               ,config=config,scoreInMI_max=with_scores,
               path_to_img=path_to_img,path_data=path_data,classes=classes,
               verbose=verbose,thresh_evaluation=thresh_evaluation,TEST_NMS=TEST_NMS,all_boxes=all_boxes
               ,PlotRegions=PlotRegions,cachefile_model_base=cachefile_model_base,number_im=np.inf,dim_rois=dim_rois,
               usecache=usecache_eval,k_per_bag=k_per_bag,num_features=num_features)
    
    for j,classe in enumerate(classes):
        AP = average_precision_score(true_label_all_test[:,j],predict_label_all_test[:,j],average=None)
        print("MI_Max version Average Precision for",classes[j]," = ",AP)
        test_precision = precision_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        test_recall = recall_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        F1 = f1_score(true_label_all_test[:,j],labels_test_predited[:,j],)
        print("Test on all the data precision = {0:.2f}, recall = {1:.2f},F1 = {2:.2f}".format(test_precision,test_recall,F1))
        precision_at_k = ranking_precision_score(np.array(true_label_all_test[:,j]), predict_label_all_test[:,j],20)
        P20_per_class += [precision_at_k]
        AP_per_class += [AP]
        R_per_class += [test_recall]
        P_per_class += [test_precision] 
    
   
    with open(cachefile_model, 'wb') as f:
        pickle.dump(name_milsvm, f)
    
    # Detection evaluation
    if database in ['watercolor','clipart','PeopleArt','IconArt_v1']:
        det_file = os.path.join(cachefilefolder, 'detections_aux.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        max_per_image = 100
        num_images_detect = len(imdb.image_index)  # We do not have the same number of images in the WikiTenLabels or IconArt_v1 case
        all_boxes_order = [[[] for _ in range(num_images_detect)] for _ in range(imdb.num_classes)]
        number_im = 0
        name_all_test = name_all_test.astype(str)
        for i in range(num_images_detect):
            name_img = imdb.image_path_at(i)
            if database=='PeopleArt':
                name_img_wt_ext = name_img.split('/')[-2] +'/' +name_img.split('/')[-1]
                name_img_wt_ext_tab =name_img_wt_ext.split('.')
                name_img_wt_ext = '.'.join(name_img_wt_ext_tab[0:-1])
            else:
                name_img_wt_ext = name_img.split('/')[-1]
                name_img_wt_ext =name_img_wt_ext.split('.')[0]
            name_img_ind = np.where(np.array(name_all_test)==name_img_wt_ext)[0]
            print(name_img_ind)
            if len(name_img_ind)==0:
                print('len(name_img_ind), images not found in the all_boxes')
                print(name_img_wt_ext)
                raise(Exception)
            else:
                number_im += 1 
            #print(name_img_ind[0])
            for j in range(1, imdb.num_classes):
                j_minus_1 = j-1
                all_boxes_order[j][i]  = all_boxes[j_minus_1][name_img_ind[0]]
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes_order[j][i][:, -1]
                            for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes_order[j][i][:, -1] >= image_thresh)[0]
                        all_boxes_order[j][i] = all_boxes_order[j][i][keep, :]
        assert (number_im==num_images_detect) # To check that we have the all the images in the detection prediction
        det_file = os.path.join(cachefilefolder, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes_order, f, pickle.HIGHEST_PROTOCOL)
        output_dir = os.path.join(cachefilefolder,'tmp',database+'_mAP.txt')
        aps =  imdb.evaluate_detections(all_boxes_order, output_dir)
        apsAt05 = aps
        print("Detection score (thres = 0.5): ",database,'with MI_Max with score =',with_scores)
        print(arrayToLatex(aps,per=True))
        ovthresh_tab = [0.3,0.1,0.]
        for ovthresh in ovthresh_tab:
            aps = imdb.evaluate_localisation_ovthresh(all_boxes_order, output_dir,ovthresh)
            if ovthresh == 0.1:
                apsAt01 = aps
            print("Detection score with thres at ",ovthresh,'with MI_Max with score =',with_scores)
            print(arrayToLatex(aps,per=True))
       
    print('~~~~~~~~')        
    print("mean Average Precision Classification for all the data = {0:.3f}".format(np.mean(AP_per_class)))    
    print("mean Precision Classification for all the data = {0:.3f}".format(np.mean(P_per_class)))  
    print("mean Recall Classification for all the data = {0:.3f}".format(np.mean(R_per_class)))  
    print("mean Precision Classification @ 20 for all the data = {0:.3f}".format(np.mean(P20_per_class)))  
    print('Mean Average Precision Classification with MI_Max with score =',with_scores,' : ')
    print(AP_per_class)
    print(arrayToLatex(AP_per_class,per=True))

    return(apsAt05,apsAt01,AP_per_class)
  
def get_tensor_by_nameDescendant(graph,name):
    """
    This function is a very bad way to get the tensor by name from the graph
    because it will test the different possibility in a ascending way starting 
    by none and stop when it get the highest
    """
    complet_name = name + ':0'
    tensor = graph.get_tensor_by_name(complet_name)
    for i in range(100):
        try:
            complet_name = name + '_'+str(i+1)+':0'
            tensor = graph.get_tensor_by_name(complet_name)
        except KeyError:
            return(tensor)
    print("We only test the 100 possible tensor, we will return the 101st tensor")
    return(tensor)

def tfR_evaluation_parall(database,num_classes,
               export_dir,dict_name_file,mini_batch_size,config,scoreInMI_max,
               path_to_img,path_data,classes,verbose,
               thresh_evaluation,TEST_NMS,all_boxes=None,PlotRegions=False,
               cachefile_model_base='',number_im=np.inf,dim_rois=5,
               usecache=True,k_per_bag=300,num_features=2048):
     """
     @param : number_im : number of image plot at maximum

     """
    
     if verbose: print('thresh_evaluation',thresh_evaluation,'TEST_NMS',TEST_NMS)
     
     thresh = thresh_evaluation # Threshold score or distance MI_max
     #TEST_NMS = 0.7 # Recouvrement entre les classes
     load_model = False
     with_tanh=True

     if PlotRegions :
         extensionStocha =  os.path.join(cachefile_model_base ,'ForIllustraion')
         path_to_output2  = os.path.join(path_data, 'tfMI_maxRegion',database,extensionStocha)
         path_to_output2_bis = os.path.join(path_to_output2, 'Train')
         path_to_output2_ter = os.path.join(path_to_output2, 'Test')
         pathlib.Path(path_to_output2_bis).mkdir(parents=True, exist_ok=True) 
         pathlib.Path(path_to_output2_ter).mkdir(parents=True, exist_ok=True)
         
     listexportsplit= export_dir.split('/')[:-1]
     export_dir_path = os.path.join(*listexportsplit)
     name_model_meta = export_dir + '.meta'
     
     get_roisScore = scoreInMI_max

     if PlotRegions:
        index_im = 0
        if verbose: print("Start ploting Regions selected by the MI_max in training phase")
        train_dataset = tf.data.TFRecordDataset(dict_name_file['trainval'])
        train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r, \
            num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,\
            num_rois=k_per_bag,dim_rois=dim_rois))
        dataset_batch = train_dataset.batch(mini_batch_size)
        if usecache:
            dataset_batch.cache()
        iterator = dataset_batch.make_one_shot_iterator()
        next_element = iterator.get_next()
        
        with tf.Session(config=config) as sess:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            load_model = True
            graph= tf.get_default_graph()
            X = get_tensor_by_nameDescendant(graph,"X")
            y = get_tensor_by_nameDescendant(graph,"y")
            if scoreInMI_max: 
                scores_tf = get_tensor_by_nameDescendant(graph,"scores")
                Prod_best = get_tensor_by_nameDescendant(graph,"ProdScore")
            else:
                Prod_best =  get_tensor_by_nameDescendant(graph,"Prod")
            if with_tanh:
                if verbose: print('use of tanh')
                Tanh = tf.tanh(Prod_best)
                mei = tf.argmax(Tanh,axis=2)
                score_mei = tf.reduce_max(Tanh,axis=2)
            else:
                mei = tf.argmax(Prod_best,axis=2)
                score_mei = tf.reduce_max(Prod_best,axis=2)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            while True:
                try:
                    next_element_value = sess.run(next_element)
                    if not(scoreInMI_max):
                        fc7s,roiss, labels,name_imgs = next_element_value
                    else:
                        fc7s,roiss,rois_scores,labels,name_imgs = next_element_value
                    if scoreInMI_max:
                        feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                    else:
                        feed_dict_value = {X: fc7s, y: labels}
                    if with_tanh:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll =\
                        sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)
                    else:
                        PositiveRegions,get_PositiveRegionsScore,PositiveExScoreAll = \
                        sess.run([mei,score_mei,Prod_best], feed_dict=feed_dict_value)

                    print('Start plotting Training exemples')
                    for k in range(len(labels)):
                        if index_im > number_im:
                            continue
                        if database in ['IconArt_v1','clipart','watercolor','PeopleArt']:
                            name_img = str(name_imgs[k].decode("utf-8") )
                        else:
                            name_img = name_imgs[k]
                        rois = roiss[k,:]
                        #if verbose: print(name_img)
                        if database in ['IconArt_v1','clipart','watercolor','PeopleArt']:
                            complet_name = os.path.join(path_to_img, name_img + '.jpg')
                            name_sans_ext = name_img
                        else:
                            name_sans_ext = os.path.splitext(name_img)[0]
                            complet_name = os.path.join(path_to_img,name_sans_ext + '.jpg')
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        scores_all = PositiveExScoreAll[:,k,:]
                        roi = roiss[k,:]
                        if dim_rois==5:
                            roi_boxes =  roi[:,1:5] / im_scales[0] 
                        else:
                            roi_boxes =  roi / im_scales[0] 
                        roi_boxes_and_score = None
                        local_cls = []
                        for j in range(num_classes):
                            if labels[k,j] == 1:
                                local_cls += [classes[j]]
                                roi_with_object_of_the_class = PositiveRegions[j,k] % len(rois) # Because we have repeated some rois
                                roi = rois[roi_with_object_of_the_class,:]
                                roi_scores = [get_PositiveRegionsScore[j,k]]
                                if dim_rois==5:
                                    roi_boxes =  roi[1:5] / im_scales[0]
                                else:
                                    roi_boxes =  roi / im_scales[0]   
                                roi_boxes_score = np.expand_dims(np.expand_dims(np.concatenate((roi_boxes,roi_scores)),axis=0),axis=0)
                                if roi_boxes_and_score is None:
                                    roi_boxes_and_score = roi_boxes_score
                                else:
                                    roi_boxes_and_score= \
                                    np.vstack((roi_boxes_and_score,roi_boxes_score))

                        cls = local_cls
                        if roi_boxes_and_score is None:
                            roi_boxes_and_score = []
                             
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                        name_output = os.path.join(path_to_output2,'Train',name_sans_ext + '_Regions.jpg')
                        if database=='PeopleArt':
                            list_split = name_output.split('/')[0:-1]
                            path_tmp = os.path.join(*list_split)
                            pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                        plt.savefig(name_output)
                        plt.close()
                        index_im +=1
                except tf.errors.OutOfRangeError:
                    break
   
     if verbose: print("Testing Time")            
     # Testing time !
     train_dataset = tf.data.TFRecordDataset(dict_name_file['test'])
     train_dataset = train_dataset.map(lambda r: parser_w_rois_all_class(r,\
        num_classes=num_classes,with_rois_scores=get_roisScore,num_features=num_features,
        num_rois=k_per_bag,dim_rois=dim_rois))
     dataset_batch = train_dataset.batch(mini_batch_size)
     if usecache:
         dataset_batch.cache()
     iterator = dataset_batch.make_one_shot_iterator()
     next_element = iterator.get_next()
     true_label_all_test =  []
     predict_label_all_test =  []
     name_all_test =  []
     FirstTime= True
     i = 0
     ii = 0
     with tf.Session(config=config) as sess:
        if load_model==False:
            new_saver = tf.train.import_meta_graph(name_model_meta)
            new_saver.restore(sess, tf.train.latest_checkpoint(export_dir_path))
            graph= tf.get_default_graph()
            X = get_tensor_by_nameDescendant(graph,"X")
            y = get_tensor_by_nameDescendant(graph,"y")
            if scoreInMI_max: 
                scores_tf = get_tensor_by_nameDescendant(graph,"scores")
                Prod_best = get_tensor_by_nameDescendant(graph,"ProdScore")
            else:
                Prod_best =  get_tensor_by_nameDescendant(graph,"Prod")
            if with_tanh:
                if verbose: print('We add the tanh in the test fct to get score between -1 and 1.')
                Tanh = tf.tanh(Prod_best)
                mei = tf.argmax(Tanh,axis=2)
                score_mei = tf.reduce_max(Tanh,axis=2)
            else:
                mei = tf.argmax(Prod_best,axis=-1)
                score_mei = tf.reduce_max(Prod_best,axis=-1)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        # Evaluation Test : Probleme ici souvent 
        while True:
            try:
                if not(scoreInMI_max):
                    fc7s,roiss, labels,name_imgs = sess.run(next_element)
                else:
                    fc7s,roiss,rois_scores,labels,name_imgs = sess.run(next_element)
                if scoreInMI_max:
                    feed_dict_value = {X: fc7s,scores_tf: rois_scores, y: labels}
                else:
                    feed_dict_value = {X: fc7s, y: labels}
                if with_tanh:
                    PositiveRegions,get_RegionsScore,PositiveExScoreAll =\
                    sess.run([mei,score_mei,Tanh], feed_dict=feed_dict_value)

                true_label_all_test += [labels]
                predict_label_all_test +=  [get_RegionsScore] # For the classification task
              
                for k in range(len(labels)):
                    if database in ['IconArt_v1','watercolor','clipart','PeopleArt']:
                        complet_name = os.path.join(path_to_img,str(name_imgs[k].decode("utf-8")) + '.jpg')
                    else:
                        complet_name = os.path.join(path_to_img,name_imgs[k] + '.jpg')
                    im = cv2.imread(complet_name)
                    blobs, im_scales = get_blobs(im)
                    scores_all = PositiveExScoreAll[:,k,:]
                   
                    roi = roiss[k,:]
                    if dim_rois==5:
                        roi_boxes =  roi[:,1:5] / im_scales[0] 
                    else:
                        roi_boxes =  roi / im_scales[0]
                  
                    for j in range(num_classes):
                        scores = scores_all[j,:]
                        inds = np.where(scores > thresh)[0]
                        cls_scores = scores[inds]
                        cls_boxes = roi_boxes[inds,:]
                        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)

                        keep = nms(cls_dets, TEST_NMS)
                        cls_dets = cls_dets[keep, :]
                        
                        all_boxes[j][i] = cls_dets
                    i+=1
    
                for l in range(len(name_imgs)): 
                    if database in ['IconArt_v1','watercolor','clipart','PeopleArt']:
                        name_all_test += [[str(name_imgs[l].decode("utf-8"))]]
                    else:
                        name_all_test += [[name_imgs[l]]]
                
                if PlotRegions:
                    if verbose and (ii%1000==0):
                        print("Plot the images :",ii)
                    if verbose and FirstTime: 
                        FirstTime = False
                        print("Start ploting Regions on test set")
                    for k in range(len(labels)):   
                        if ii > number_im:
                            continue
                        if  database in ['IconArt_v1','watercolor','clipart','PeopleArt']:
                            name_img = str(name_imgs[k].decode("utf-8") )
                        else:
                            name_img = name_imgs[k]
                        rois = roiss[k,:]
                        if database in ['IconArt_v1','watercolor','clipart','PeopleArt']:
                            complet_name = os.path.join(path_to_img,name_img + '.jpg')
                            name_sans_ext = name_img
                        elif(database=='Wikidata_Paintings') or (database=='Wikidata_Paintings_miniset_verif'):
                            name_sans_ext = os.path.splitext(name_img)[0]
                            complet_name = os.path.join(path_to_img,name_sans_ext + '.jpg')
                        
                        im = cv2.imread(complet_name)
                        blobs, im_scales = get_blobs(im)
                        roi_boxes_and_score = []
                        local_cls = []
                        for j in range(num_classes):
                            
                            cls_dets = all_boxes[j][ii] # Here we have #classe x box dim + score
                            if len(cls_dets) > 0:
                                local_cls += [classes[j]]
                                roi_boxes_score = cls_dets
                                if roi_boxes_and_score is None:
                                    roi_boxes_and_score = [roi_boxes_score]
                                else:
                                    roi_boxes_and_score += [roi_boxes_score] 

                        if roi_boxes_and_score is None: roi_boxes_and_score = [[]]
                        ii += 1    
                        cls = local_cls
                        # In this case we will plot several version of the image
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=-np.inf)
                        name_output = os.path.join(path_to_output2,'Test',name_sans_ext + '_Regions.jpg')
                        if database=='PeopleArt':
                            list_split = name_output.split('/')[0:-1]
                            path_tmp = os.path.join(*list_split)
                            pathlib.Path(path_tmp).mkdir(parents=True, exist_ok=True) 
                        plt.savefig(name_output)
                        plt.close()
                        
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.25)
                        name_output = os.path.join(path_to_output2,'Test',name_sans_ext + '_Regions_over025.jpg')
                        plt.savefig(name_output)
                        plt.close()
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.5)
                        name_output = os.path.join(path_to_output2,'Test' , name_sans_ext + '_Regions_over05.jpg')
                        plt.savefig(name_output)
                        plt.close()
                        vis_detections_list(im, cls, roi_boxes_and_score, thresh=0.75)
                        name_output = os.path.join(path_to_output2,'Test',name_sans_ext + '_Regions_over075.jpg')
                        plt.savefig(name_output)
                        plt.close()
                       
            except tf.errors.OutOfRangeError:
                break
     tf.reset_default_graph()
     true_label_all_test = np.concatenate(true_label_all_test)
     predict_label_all_test = np.transpose(np.concatenate(predict_label_all_test,axis=1))
     name_all_test = np.concatenate(name_all_test)
     labels_test_predited = (np.sign(predict_label_all_test) +1.)/2
     labels_test_predited[np.where(labels_test_predited==0.5)] = 0 # To deal with the case where predict_label_all_test == 0 
     return(true_label_all_test,predict_label_all_test,name_all_test,
            labels_test_predited,all_boxes)

 
