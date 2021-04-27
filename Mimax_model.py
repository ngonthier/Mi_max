#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 09:47:54 2018

@author: said ladjal and gonthier
"""

import tensorflow as tf
import numpy as np
npt=np.float32
tft=np.float32
import time
import multiprocessing
import os
import pathlib

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_feature_reshape(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value.reshape(-1)))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def test_version_sup(version_str):
    version_str_tab = version_str.split('.')
    tf_version_teb =  tf.__version__.split('.')
    status = False
    for a,b in zip(tf_version_teb,version_str_tab):
        if float(a) >= float(b):
            status = True
    return(status)

class tf_MI_max():
    """
    The MI_Max model proposed in Weakly Supervised Object Detection in Artworks
    https://arxiv.org/abs/1810.02569
    
    This function also contains the Polyhedral Mimax model proposed in 
    Multiple instance learning on deep features for weakly supervised object detection with extreme domain shifts
    https://arxiv.org/abs/2008.01178
    """

    def __init__(self,LR=0.01,C=1.0,restarts=0, max_iters=300,
                 verbose=True,Optimizer='GradientDescent',mini_batch_size=200,
                 buffer_size=10000,num_features=2048,
                  num_rois=300,num_classes=10,loss_type='',
                  is_betweenMinus1and1=False,CV_Mode=None,num_split=2,with_scores=False,
                  epsilon=0.0,usecache=True,normalizeW=False,
                  Polyhedral=False): 
        """
        @param LR : Learning rate : pas de gradient de descente [default: 0.01]
        @param C : the loss/regularization tradeoff constant [default: 1.0]
        @param C_finalSVM : the loss/regularization  term fo the final SVM training [default: 1.0]
        @param restarts : the number of random restarts [default: 0]
        @param max_iters : the maximum number of iterations in the inter loop of
                           the optimization procedure [default: 300]
        @param symway : If positive and negative bag are treated as the same way 
            or not [default: True]
        @param all_notpos_inNeg : All the element of the positive bag that are 
            not positive element are put in the negative class [default: True]
        @param gridSearch :GridSearch of the final SVC classifier [default: False]
        @param n_jobs : number of parallel jobs during the gridsearch -1 means = 
            number of cores  [default: -1]
        @param final_clf : final classifier used after the determination of the 
            element [choice : defaultSGD, linearSVC] [default : linearSVC]
        @param verbose : print optimization status messages [default: True]
        @param Optimizer : possible : 'GradientDescent','Momentum','Adam','Adagrad','lbfgs'
        @param mini_batch_size : taille des mini_batch_size
        @param buffer_size : taille du buffer
        @param loss_type : Type of the loss :
            '' or None : the original loss function from our work, a kind of hinge loss but with a prediction between -1 and 1
            'MSE' : Mean squared error
            'hinge_tanh' : hinge loss on the tanh output (it seems to be the same as the original loss)
            'hinge': hinge loss without the tanh : normal way to use it
            'log' : the classification log loss : kind of crossentropy loss
        @param num_features : pnumbre de features
        @param num_rois : nombre de regions d interet
        @param num_classes : numbre de classes dans la base
        @param max_iters_sgdc : Nombre d iterations pour la descente de gradient stochastique classification
        @param debug : default False : if we want to debug 
        @param is_betweenMinus1and1 : default False : if we have the label value alreaddy between -1 and 1
        @param CV_Mode : default None : cross validation mode in the MI_max : 
            Choice : None, 'CV' in k split 
        @param num_split : default 2 : the number of split/fold used in the cross validation method
        @param with_scores : default False : Multiply the scalar product before the max by the objectness score from the FasterRCNN
        @param epsilon : default 0. : The term we add to the object score
        @param normalizeW : normalize the W vectors before optimization
        @param : Polyhedral use the max of the max of product and keep all the (W,b) learnt
            in order to have a polyhedral model
            (default False)
        """
        self.LR = LR
        self.C = C
        self.restarts = restarts
        self.max_iters = max_iters
        self.loss_type = loss_type
        self.verbose = verbose
        self.Optimizer = Optimizer
        self.mini_batch_size = mini_batch_size
        self.buffer_size = buffer_size
        self.num_features = num_features
        self.num_rois = num_rois
        self.num_classes = num_classes
        self.is_betweenMinus1and1 = is_betweenMinus1and1
        self.np_pos_value = 1
        self.np_neg_value = 1 # Those elements will be replace by matrix if the dataset contains several classes
        self.CV_Mode = CV_Mode
        if not(CV_Mode is None):
            if not(CV_Mode in ['CV','']):
                print(CV_Mode,' is unknwonw')
                raise(NotImplementedError)
            assert(num_split>1) # Il faut plus d un folder pour separer
            self.num_split = num_split # Only useful if CrossVal==True
            if num_split>2:
                print('The use of more that 2 spits seem to slow a lot the computation with the use of shard')
        self.with_scores = with_scores 
        self.epsilon = epsilon# Used to avoid having a zero score
        self.Cbest =None
        # case of Cvalue
        self.C_values =  np.arange(0.5,1.5,0.1,dtype=np.float32) # Case used in VISART2018
        self.usecache = usecache
        self.normalizeW = normalizeW
        self.Polyhedral = Polyhedral
        
    def parser(self,record):

        keys_to_features={
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        #tf.Print(label,[label])
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7, label
    
    def parser_wRoiScore(self,record):
        keys_to_features={
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        roi_scores = parsed['roi_scores']
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7,roi_scores, label
    
    def parser_all_classes(self,record):
        keys_to_features={
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7, label
    
    def parser_all_classes_wRoiScore(self,record):
        keys_to_features={
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)}
        parsed = tf.parse_single_example(record, keys_to_features)
        
        label = parsed['label']
        roi_scores = parsed['roi_scores']
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        return fc7,roi_scores, label
    
    def parser_w_rois(self,record):
        # Perform additional preprocessing on the parsed data.
        keys_to_features={
                    'num_regions':  tf.FixedLenFeature([], tf.int64),
                    'num_features':  tf.FixedLenFeature([], tf.int64),
                    'rois': tf.FixedLenFeature([5*self.num_rois],tf.float32), # Here we can have a problem if the rois is not size 5
                    'roi_scores':tf.FixedLenFeature([self.num_rois],tf.float32),
                    'fc7': tf.FixedLenFeature([self.num_rois*self.num_features],tf.float32),
                    'label' : tf.FixedLenFeature([self.num_classes],tf.float32)
                    }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        # Cast label data into int32
        label = parsed['label']
        name_img = parsed['name_img']
        label = tf.slice(label,[self.class_indice],[1])
        label = tf.squeeze(label) # To get a vector one dimension
        fc7 = parsed['fc7']
        fc7 = tf.reshape(fc7, [self.num_rois,self.num_features])
        rois = parsed['rois']
        rois = tf.reshape(rois, [self.num_rois,5])           
        return fc7,rois, label,name_img
    
    def tf_dataset_use_per_batch(self,train_dataset,performance=False):
        
        if test_version_sup('1.6') and performance:
            dataset_batch = train_dataset.apply(tf.contrib.data.map_and_batch(
                map_func=self.first_parser, batch_size=self.mini_batch_size))
        else:
            train_dataset = train_dataset.map(self.first_parser,
                                          num_parallel_calls=self.cpu_count)
            dataset_batch = train_dataset.batch(self.mini_batch_size)
        if self.usecache:
            dataset_batch = dataset_batch.cache()
        dataset_batch = dataset_batch.prefetch(self.mini_batch_size)
        iterator_batch = dataset_batch.make_initializable_iterator()
        return(iterator_batch)
    
    def eval_loss(self,sess,iterator_batch,loss_batch):
        if self.class_indice==-1:
            if self.restarts_paral_V2:
                loss_value = np.zeros((self.paral_number_W*self.num_classes,),dtype=np.float32)
                if self.MaxOfMax or self.MaxMMeanOfMax or self.MaxTopMinOfMax:
                    loss_value = np.zeros((self.num_classes,),dtype=np.float32)
            elif self.restarts_paral_Dim:
                loss_value = np.zeros((self.paral_number_W,self.num_classes),dtype=np.float32)
        else:
            loss_value = np.zeros((self.paral_number_W,),dtype=np.float32)
        sess.run(iterator_batch.initializer)
        while True:
            try:
                loss_value_tmp = sess.run(loss_batch)
                loss_value += loss_value_tmp
                break
            except tf.errors.OutOfRangeError:
                break
        return(loss_value)
        
    def fit_MI_max_tfrecords(self,data_path,C_Searching=False,shuffle=True):
        """" 
        This function run per batch on the tfrecords data folder
        @param : data_path : 
        @param : choose of the class to run the optimisation on, if == -1 , then 
        run all the class at once
        @param : shuffle or not the dataset 
        """
        
        self.C_Searching= C_Searching

        if self.C_Searching:
            C_values = self.C_values
            self.Cbest = np.zeros((self.num_classes,))
            self.paral_number_W = self.restarts +1
            C_value_repeat = np.repeat(C_values,repeats=(self.paral_number_W*self.num_classes),axis=0)
            self.paral_number_W *= len(C_values)
            if self.verbose: print('We will compute :',len(C_value_repeat),'W vectors due to the C searching')
        else:
            self.paral_number_W = self.restarts +1
        
        ## Debut de la fonction        
        self.cpu_count = multiprocessing.cpu_count()
        train_dataset_init = tf.data.TFRecordDataset(data_path)
        
        if self.CV_Mode=='CV':
            if self.verbose: print('Use of the Cross Validation with ',self.num_split,' splits')
            train_dataset_tmp = train_dataset_init.shard(self.num_split,0)
            for i in range(1,self.num_split-1):
                train_dataset_tmp2 = train_dataset_init.shard(self.num_split,i)
                train_dataset_tmp = train_dataset_tmp.concatenate(train_dataset_tmp2)
            train_dataset = train_dataset_tmp
        else:
            train_dataset = train_dataset_init
            # The second argument is the index of the subset used
        if self.with_scores:
            self.first_parser = self.parser_all_classes_wRoiScore
        else:
            self.first_parser = self.parser_all_classes
          
        iterator_batch = self.tf_dataset_use_per_batch(train_dataset)
        
        if self.with_scores:
            X_batch,scores_batch, label_batch = iterator_batch.get_next()
        else:
            X_batch, label_batch = iterator_batch.get_next()
        # Calcul preliminaire a la definition de la fonction de cout 
        self.config = tf.ConfigProto()
        self.config.intra_op_parallelism_threads = 16
        self.config.inter_op_parallelism_threads = 16
        self.config.gpu_options.allow_growth = True
        
        minus_1 = tf.constant(-1.)

        label_vector = tf.placeholder(tf.float32, shape=(None,self.num_classes))
        if self.is_betweenMinus1and1:
            add_np_pos = tf.divide(tf.reduce_sum(tf.add(label_vector,tf.constant(1.))),tf.constant(2.))
            add_np_neg = tf.divide(tf.reduce_sum(tf.add(label_vector,minus_1)),tf.constant(-2.))
        else:
            add_np_pos = tf.reduce_sum(label_vector,axis=0)
            add_np_neg = -tf.reduce_sum(tf.add(label_vector,minus_1),axis=0)
        np_pos_value = np.zeros((self.num_classes,),dtype=np.float32)
        np_neg_value = np.zeros((self.num_classes,),dtype=np.float32)
       
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(iterator_batch.initializer)
            while True:
              try:
                  # Attention a chaque fois que l on appelle la fonction iterator on avance
                  label_batch_value = sess.run(label_batch)
                  np_pos_value += sess.run(add_np_pos, feed_dict = {label_vector:label_batch_value})
                  np_neg_value += sess.run(add_np_neg, feed_dict = {label_vector:label_batch_value})
              except tf.errors.OutOfRangeError:
                break
            
        self.np_pos_value = np_pos_value
        self.np_neg_value = np_neg_value
        if self.verbose:print("Finished to compute the proportion of each label :",np_pos_value,np_neg_value)
       
        if self.CV_Mode=='CV':
            train_dataset_tmp = train_dataset_init.shard(self.num_split,0)
            for i in range(1,self.num_split-1):
                train_dataset_tmp2 = train_dataset.shard(self.num_split,i)
                train_dataset_tmp = train_dataset_tmp.concatenate(train_dataset_tmp2)
            train_dataset2 = train_dataset_tmp
            train_dataset = train_dataset_init.shard(self.num_split,self.num_split-1) 
            # The last fold is keep for doing the cross validation
            iterator_batch = self.tf_dataset_use_per_batch(train_dataset)
            if self.with_scores or self.seuillage_by_score or self.obj_score_add_tanh  or self.obj_score_mul_tanh:
                X_batch,scores_batch, label_batch = iterator_batch.get_next()
            else:
                X_batch, label_batch = iterator_batch.get_next() 
        else:
            # TODO test !
            train_dataset2 = tf.data.TFRecordDataset(data_path) # train_dataset_init ?  A tester
        
        train_dataset2 = train_dataset2.map(self.first_parser,
                                        num_parallel_calls=self.cpu_count)
        if shuffle:
            dataset_shuffle = train_dataset2.shuffle(buffer_size=self.buffer_size,
                                                     reshuffle_each_iteration=True) 
        else:
            dataset_shuffle = train_dataset2
        dataset_shuffle = dataset_shuffle.batch(self.mini_batch_size)
        if self.usecache:
            dataset_shuffle = dataset_shuffle.cache() 
        dataset_shuffle = dataset_shuffle.repeat() 
        dataset_shuffle = dataset_shuffle.prefetch(self.mini_batch_size) 
        shuffle_iterator = dataset_shuffle.make_initializable_iterator()
        if self.with_scores:
            X_,scores_, y_ = shuffle_iterator.get_next()
        else:
            X_, y_ = shuffle_iterator.get_next()

        # Definition of the graph 
        W=tf.Variable(tf.random_normal([self.paral_number_W*self.num_classes,self.num_features], stddev=1.),name="weights")
        b=tf.Variable(tf.random_normal([self.paral_number_W*self.num_classes,1,1], stddev=1.), name="bias")
        if test_version_sup('1.8'):
            normalize_W = W.assign(tf.nn.l2_normalize(W,axis=0)) 
        else:
            normalize_W = W.assign(tf.nn.l2_normalize(W,dim=0)) 
        W_r=W

        Prod = tf.einsum('ak,ijk->aij',W_r,X_)
        Prod=tf.add(Prod,b)
           
        if self.with_scores: 
            if self.verbose: print('With score multiplication')
            Prod=tf.multiply(Prod,tf.add(scores_,self.epsilon))

        Max=tf.reduce_max(Prod,axis=-1) 

        if self.is_betweenMinus1and1:
            weights_bags_ratio = -tf.divide(tf.add(y_,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(y_,-1.),tf.multiply(-2.,np_neg_value))
            # Need to add 1 to avoid the case 
            # The wieght are negative for the positive exemple and positive for the negative ones !!!
        else:
            weights_bags_ratio = -tf.divide(y_,np_pos_value) + tf.divide(-tf.add(y_,-1),np_neg_value)

        weights_bags_ratio = tf.tile(tf.transpose(weights_bags_ratio,[1,0]),[self.paral_number_W,1])
        y_long_pm1 = tf.tile(tf.transpose(tf.add(tf.multiply(y_,2),-1),[1,0]), [self.paral_number_W,1])
           
        y_tilde_i = tf.tanh(Max)
        if self.loss_type == '' or self.loss_type is None:
            Tan= tf.reduce_sum(tf.multiply(y_tilde_i,weights_bags_ratio),axis=-1) # Sum on all the positive exemples 
        elif self.loss_type=='hinge':
            if self.verbose: print('Used of the hinge loss without tanh')
            hinge = tf.maximum(tf.add(-tf.multiply(Max,y_long_pm1),1.),0.)
            Tan = tf.reduce_sum(tf.multiply(hinge,tf.abs(weights_bags_ratio)),axis=-1)

        if self.C_Searching:    
            loss= tf.add(Tan,tf.multiply(C_value_repeat,tf.reduce_sum(tf.pow(W_r,2),axis=-1)))
        else:
            W_r_reduce = tf.reduce_sum(tf.pow(W_r,2),axis=-1)
            loss= tf.add(Tan,tf.multiply(self.C,W_r_reduce))
                        
        #Definition on batch
        Prod_batch = tf.einsum('ak,ijk->aij',W_r,X_batch)
        Prod_batch=tf.add(Prod_batch,b)
           
        if self.with_scores: 
            Prod_batch=tf.multiply(Prod_batch,tf.add(scores_batch,self.epsilon))
           
        Max_batch=tf.reduce_max(Prod_batch,axis=-1) # We take the max because we have at least one element of the bag that is positive

        if self.is_betweenMinus1and1:
            weights_bags_ratio_batch = -tf.divide(tf.add(label_batch,1.),tf.multiply(2.,np_pos_value)) + tf.divide(tf.add(label_batch,-1.),tf.multiply(-2.,np_neg_value))
            # Need to add 1 to avoid the case 
            # The wieght are negative for the positive exemple and positive for the negative ones !!!
        else:
            weights_bags_ratio_batch = -tf.divide(label_batch,np_pos_value) + tf.divide(-tf.add(label_batch,-1),np_neg_value) # Need to add 1 to avoid the case 
        weights_bags_ratio_batch = tf.tile(tf.transpose(weights_bags_ratio_batch,[1,0]),[self.paral_number_W,1])
        y_long_pm1_batch =  tf.tile(tf.transpose(tf.add(tf.multiply(label_batch,2),-1),[1,0]), [self.paral_number_W,1])
    
        y_tilde_i_batch = Max_batch
                
        if self.loss_type == '' or self.loss_type is None:
            Tan_batch= tf.reduce_sum(tf.multiply(y_tilde_i_batch,weights_bags_ratio_batch),axis=-1) # Sum on all the positive exemples 
        elif self.loss_type=='hinge':
            hinge_batch = tf.maximum(tf.add(-tf.multiply(Max_batch,y_long_pm1_batch),1.),0.)
            Tan_batch = tf.reduce_sum(tf.multiply(hinge_batch,tf.abs(weights_bags_ratio_batch)),axis=-1)
            
        loss_batch= Tan_batch
           
        if(self.Optimizer == 'GradientDescent'):
            optimizer = tf.train.GradientDescentOptimizer(self.LR) 
        elif(self.Optimizer == 'Momentum'):
            if self.optimArg is None:
                optimizer = tf.train.MomentumOptimizer(self.LR,0.9) 
            else:
                optimizer = tf.train.MomentumOptimizer(self.optimArg['learning_rate'],self.optimArg['momentum']) 
        elif(self.Optimizer == 'Adam'):
            if self.optimArg is None:
                optimizer = tf.train.AdamOptimizer(self.LR) 
                # Default value  : beta1=0.9,beta2=0.999,epsilon=1e-08, 
                # maybe epsilon should be 0.1 or 1 cf https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=\
                self.optimArg['learning_rate'],beta1=self.optimArg['beta1'],\
                beta2=self.optimArg['beta2'],epsilon=self.optimArg['epsilon'])
        elif(self.Optimizer == 'Adagrad'):
            optimizer = tf.train.AdagradOptimizer(self.LR) 
        elif not(self.Optimizer == 'lbfgs'):
            print("The optimizer is unknown",self.Optimizer)
            raise(NotImplementedError)
            
        if self.Optimizer in ['GradientDescent','Momentum','Adam','Adagrad']:
            train = optimizer.minimize(loss)  
        
        sess = tf.Session(config=self.config)     
        init_op = tf.group(tf.global_variables_initializer()\
                           ,tf.local_variables_initializer())
            
        if self.verbose : 
            print('Start with the restarts',self.restarts,' in parallel')
            t0 = time.time()

        sess.run(init_op)
        sess.run(shuffle_iterator.initializer)
        if self.normalizeW:
            sess.run(normalize_W) # Normalize the W vector
        
        if self.Optimizer in ['GradientDescent','Momentum','Adam']:
            for step in range(self.max_iters):
                sess.run(train)
                
        elif self.Optimizer=='lbfgs':
            maxcor = 30
            optimizer_kwargs = {'maxiter': self.max_iters,'maxcor': maxcor}
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,method='L-BFGS-B',
                                                               options=optimizer_kwargs)    
            optimizer.minimize(sess)
        else:
            print("The optimizer is unknown",self.Optimizer)
            raise(NotImplementedError)
                

        loss_value = np.zeros((self.paral_number_W*self.num_classes,),dtype=np.float32)
        sess.run(iterator_batch.initializer)

        while True:
            try:
                loss_value += sess.run(loss_batch)
                break
            except tf.errors.OutOfRangeError:
                break
        

        loss_value_min = []
        W_best = np.zeros((self.num_classes,self.num_features),dtype=np.float32)
        b_best = np.zeros((self.num_classes,1,1),dtype=np.float32)
        if self.restarts>0:
            W_tmp=sess.run(W)
            b_tmp=sess.run(b)
            for j in range(self.num_classes):
                loss_value_j = loss_value[j::self.num_classes]
#                                print('loss_value_j',loss_value_j)
                argmin = np.argmin(loss_value_j,axis=0)
                loss_value_j_min = np.min(loss_value_j,axis=0)
                W_best[j,:] = W_tmp[j+argmin*self.num_classes,:]
                b_best[j,:,:] = b_tmp[j+argmin*self.num_classes]
                loss_value_min+=[loss_value_j_min]
                if (self.C_Searching) and self.verbose:
                    print('Best C values : ',C_value_repeat[j+argmin*self.num_classes],'class ',j)
                if (self.C_Searching): self.Cbest[j] = C_value_repeat[j+argmin*self.num_classes]
            self.bestloss = loss_value_min
            if self.verbose : 
                print("bestloss",loss_value_min)
                t1 = time.time()
                print("durations after simple training :",str(t1-t0),' s')
        else:
            # In the case of MaxOfMax : we keep all the vectors
            W_best=sess.run(W)
            b_best=sess.run(b)
            if self.verbose : print("loss",loss_value)
            self.bestloss = loss_value
       
        ## End we save the best w
        saver = tf.train.Saver()
        X_= tf.identity(X_, name="X")
        y_ = tf.identity(y_, name="y")
        if self.with_scores:
            scores_ = tf.identity(scores_,name="scores")

        Prod_best=tf.add(tf.einsum('ak,ijk->aij',tf.convert_to_tensor(W_best),X_)\
                     ,b_best,name='Prod')

        #Integration du score dans ce qui est retourner a la fin
        if self.with_scores: 
            Prod_score=tf.multiply(Prod_best,tf.add(scores_,self.epsilon),name='ProdScore')
           
        head, tail = os.path.split(data_path)
        export_dir_folder = os.path.join(head,'MI_max')
        pathlib.Path(export_dir_folder).mkdir(parents=True, exist_ok=True) 
        export_dir = os.path.join(export_dir_folder,str(time.time()))
        name_model = os.path.join(export_dir,'model')
        saver.save(sess,name_model)
        
        sess.close()
        if self.verbose : print("Return MI_max weights")
        return(name_model)      

    def get_PositiveRegions(self):
        return(self.PositiveRegions.copy())
     
    def get_PositiveRegionsScore(self):
        return(self.PositiveRegionsScore.copy())
     
    def get_PositiveExScoreAll(self):
        return(self.PositiveExScoreAll.copy())
        
    def get_NegativeRegions(self):
        return(self.NegativeRegions.copy())
     
    def get_NegativeRegionsScore(self):
        return(self.NegativeRegionsScore.copy()) 
    
    def get_porportions(self):
        return(self.np_pos_value,self.np_neg_value)
        
    def get_Cbest(self):
        return(self.Cbest)
        
    def set_CV_Mode(self,new_CV_Mode):
        self.CV_Mode = new_CV_Mode
        
    def set_C(self,new_C):
        self.C = new_C
        
    def get_bestloss(self):
        return(self.bestloss)
