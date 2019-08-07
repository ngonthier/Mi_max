#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:42:41 2017

Based on the Tensorflow implementation of Faster-RCNNN but it has been modified a lot
https://github.com/endernewton/tf-faster-rcnn

It is a convertion for Python 3

Faster RCNN re-scale  the  images  such  that  their  shorter  side  = 600 pixels  

@author: gonthier

You can find the weight here : https://partage.mines-telecom.fr/index.php/s/ep52PPAxSI932zY
or here https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ


"""
import tensorflow as tf
from tf_faster_rcnn.lib.nets.vgg16 import vgg16
from tf_faster_rcnn.lib.nets.resnet_v1 import resnetv1
from tf_faster_rcnn.lib.model.test import TL_im_detect
import matplotlib.pyplot as plt
import numpy as np
import os,cv2
import os.path
import pathlib
from tool_on_Regions import reduce_to_k_regions
from IMDB import get_database

CLASSESVOC = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

CLASSESCOCO = ('__background__','person', 'bicycle','car','motorcycle', 'aeroplane','bus',
               'train','truck','boat',
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
    ,'vgg16_coco': ('vgg16_faster_rcnn_iter_1190000.ckpt',)    
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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_reshape(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def vis_detections(im, class_name, dets, thresh=0.5,with_title=True,draw=True):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    if with_title:
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                      fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    if draw:
        plt.draw()
    
def vis_detections_list(im, class_name_list, dets_list, thresh=0.5,list_class=None,Correct=None):
    """Draw detected bounding boxes."""

    list_colors = ['#e6194b','#3cb44b','#ffe119','#0082c8',	'#f58231','#911eb4','#46f0f0','#f032e6',	
                   '#d2f53c','#fabebe',	'#008080','#e6beff','#aa6e28','#fffac8','#800000',
                   '#aaffc3','#808000','#ffd8b1','#000080','#808080','#FFFFFF','#000000']	
    i_color = 0
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
   
    for class_name,dets in zip(class_name_list,dets_list):
#        print(class_name,np.array(dets).shape)
        inds = np.where(dets[:, -1] >= thresh)[0]
        if not(len(inds) == 0):
            if list_class is None:
                color = list_colors[i_color]
                i_color = ((i_color + 1) % len(list_colors))
            else:
                i_color = np.where(np.array(list_class)==class_name)[0][0] % len(list_colors)
                color = list_colors[i_color]
            for i in inds:
                bbox = dets[i, :4] # Boxes are score, x1,y1,x2,y2
                score = dets[i, -1]
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=3.5) # Need (x,y) lower corner then width, height
                    )
                ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=14, color='white')
                            
    plt.axis('off')
    plt.tight_layout()
    if not (Correct is None):
        print("This have never been tested")
        # In this case, we will draw a rectangle green or red around the image
        if Correct=='Correct':
            color = 'g'
        elif Correct=='Incorrect':
            color=  'r'
        elif Correct=='Missing':
            color=  'o'
        elif Correct=='MultipleDetect':
            color=  'p'
        linewidth = 10
        x = linewidth
        y = linewidth
        h = im.shape[0] - x
        w = im.shape[1] - y
        ax.add_patch(plt.Rectangle((x,y),h,w, fill=False,
                      edgecolor=color, linewidth=linewidth)) 
    plt.draw()
    

                   
def Compute_Faster_RCNN_features(demonet='res152_COCO',nms_thresh = 0.7,database='IconArt_v1'
                                 ,verbose=True,k_regions=300
                                 ,path_data='data',path_output='output',path_to_model='models'):
    """
    @param : demonet : the backbone net used it can be 'vgg16_VOC07',
        'vgg16_VOC12','vgg16_COCO','res101_VOC12','res101_COCO','res152_COCO'
    @param : nms_thresh : the nms threshold on the Region Proposal Network
    @param : database name of the dataset
    @param : k_regions : number of region per image
    @param : path_data path to the dataset
    @param : path_output path to the output model 
    @param : path_to_model path to the pretarined model
    """
    
    item_name,path_to_img,classes,ext,num_classes,str_val,df_label = get_database(database,default_path_imdb =path_data)
    

    N=1
    extL2 = ''
    savedstr = '_all'
    layer='fc7'
    tf.reset_default_graph() # Needed to use different nets one after the other
    if verbose: print(demonet)
    if 'VOC'in demonet:
        CLASSES = CLASSES_SET['VOC']
        anchor_scales=[8, 16, 32] # It is needed for the right net architecture !! 
    elif 'COCO'in demonet:
        CLASSES = CLASSES_SET['COCO']
        anchor_scales = [4, 8, 16, 32] # we  use  3  aspect  ratios  and  4  scales (adding 64**2)
    nbClassesDemoNet = len(CLASSES)
    pathlib.Path(path_to_model).mkdir(parents=True, exist_ok=True) 
    tfmodel = os.path.join(path_to_model,NETS_Pretrained[demonet])
    if not(os.path.exists(tfmodel)):
        print("You have to download the Faster RCNN pretrained, see README")
        return(0)
    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    
    # load network
    if  'vgg16' in demonet:
      net = vgg16()
      size_output = 4096
    elif 'res101' in demonet:
      net = resnetv1(num_layers=101)
      size_output = 2048
    elif 'res152' in demonet:
      net = resnetv1(num_layers=152)
      size_output = 2048
    else:
      raise NotImplementedError

    net.create_architecture("TEST", nbClassesDemoNet,
                          tag='default', anchor_scales=anchor_scales,
                          modeTL= True,nms_thresh=nms_thresh)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    features_resnet_dict= {}
    
    sets = ['trainval','test']
   
    if k_regions==300:
        k_per_bag_str = ''
    else:
        k_per_bag_str = '_k'+str(k_regions)
    dict_writers = {}
    for set_str in sets:
        name_pkl_all_features = os.path.join(path_output,'FasterRCNN_'+ demonet +'_'+database+'_N'+str(N)+extL2+'_TLforMIL_nms_'+str(nms_thresh)+savedstr+k_per_bag_str+'_'+set_str+'.tfrecords')
        dict_writers[set_str] = tf.python_io.TFRecordWriter(name_pkl_all_features)
   
    Itera = 1000
    for i,name_img in  enumerate(df_label[item_name]):
        
        if i%Itera==0:
            if verbose : print(i,name_img)
        if database in ['IconArt_v1','watercolor']:
            complet_name = path_to_img + name_img + '.jpg'
            name_sans_ext = name_img
        elif database=='PeopleArt':
            complet_name = path_to_img + name_img
            name_sans_ext = os.path.splitext(name_img)[0]
        try:    
            im = cv2.imread(complet_name)
            height = im.shape[0]
            width = im.shape[1]
        except AttributeError:
            print(complet_name,'is missing')
            continue
        cls_score, cls_prob, bbox_pred, rois,roi_scores, fc7,pool5 = TL_im_detect(sess, net, im) # Arguments: im (ndarray): a color image in BGR order
        
        if k_regions==300:
            num_regions = fc7.shape[0]
            num_features = fc7.shape[1]
            dim1_rois = rois.shape[1]
            classes_vectors = np.zeros((num_classes,1))
            rois_tmp = np.zeros((k_regions,5))
            roi_scores_tmp = np.zeros((k_regions,1))
            fc7_tmp = np.zeros((k_regions,size_output))
            rois_tmp[0:rois.shape[0],0:rois.shape[1]] = rois
            roi_scores_tmp[0:roi_scores.shape[0],0:roi_scores.shape[1]] = roi_scores
            fc7_tmp[0:fc7.shape[0],0:fc7.shape[1]] = fc7           
            rois = rois_tmp
            roi_scores =roi_scores_tmp
            fc7 = fc7_tmp
        else:
            # We will select only k_regions 
            new_nms_thresh = 0.0
            score_threshold = 0.1
            minimal_surface = 36*36
            
            num_regions = k_regions
            num_features = fc7.shape[1]
            dim1_rois = rois.shape[1]
            classes_vectors = np.zeros((num_classes,1))
            rois_reduce,roi_scores_reduce,fc7_reduce =  reduce_to_k_regions(k_regions,rois, \
                                                   roi_scores, fc7,new_nms_thresh, \
                                                   score_threshold,minimal_surface)
            if(len(fc7_reduce) >= k_regions):
                rois = rois_reduce[0:k_regions,:]
                roi_scores =roi_scores_reduce[0:k_regions,]
                fc7 = fc7_reduce[0:k_regions,:]
            else:
                number_repeat = k_regions // len(fc7_reduce)  +1
                f_repeat = np.repeat(fc7_reduce,number_repeat,axis=0)
                roi_scores_repeat = np.repeat(roi_scores_reduce,number_repeat,axis=0)
                rois_reduce_repeat = np.repeat(rois_reduce,number_repeat,axis=0)
                rois = rois_reduce_repeat[0:k_regions,:]
                roi_scores =roi_scores_repeat[0:k_regions,]
                fc7 = f_repeat[0:k_regions,:]
        
        if database in ['watercolor','PeopleArt']:
            for j in range(num_classes):
                value = int((int(df_label[classes[j]][i])+1.)/2.)
                #print(value)
                classes_vectors[j] = value
        if database in ['IconArt_v1']:
            for j in range(num_classes):
                value = int(df_label[classes[j]][i])
                classes_vectors[j] = value
        features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'num_regions': _int64_feature(num_regions),
            'num_features': _int64_feature(num_features),
            'dim1_rois': _int64_feature(dim1_rois),
            'rois': _floats_feature(rois),
            'roi_scores': _floats_feature(roi_scores),
            'fc7': _floats_feature(fc7),
            'label' : _floats_feature(classes_vectors),
            'name_img' : _bytes_feature(str.encode(name_sans_ext))})
        example = tf.train.Example(features=features)    
        
        if database=='PeopleArt':
            if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                dict_writers['trainval'].write(example.SerializeToString())
            elif (df_label.loc[df_label[item_name]==name_img]['set']=='val').any():
                dict_writers['trainval'].write(example.SerializeToString())
            elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                dict_writers['test'].write(example.SerializeToString())
        if database in ['watercolor','IconArt_v1']\
                        or 'IconArt_v1' in database:
            if (df_label.loc[df_label[item_name]==name_img]['set']=='train').any():
                dict_writers['trainval'].write(example.SerializeToString())
            elif (df_label.loc[df_label[item_name]==name_img]['set']=='test').any():
                dict_writers['test'].write(example.SerializeToString())
        
    for set_str  in sets:
        dict_writers[set_str].close()

