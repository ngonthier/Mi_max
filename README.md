# MI_max : Multi Instance Perceptron for weakly supervised transfer learning of deep detector - [Weakly Supervised Object Detection in Artworks](https://arxiv.org/abs/1810.02569)

By [Gonhier Nicolas](https://perso.telecom-paristech.fr/gonthier/), [Gousseau Yann](https://perso.telecom-paristech.fr/gousseau/), [Ladjal Said](https://perso.telecom-paristech.fr/ladjal/) and [Bonfait Olivier](http://tristan.u-bourgogne.fr/CGC/chercheurs/Bonfait/Olivier_Bonfait.html).
**This is a Tensorflow implementation of our Mi_max model.**

**This code can be used to reproduce results on [IconArt](https://wsoda.telecom-paristech.fr/downloads/dataset/), [Watercolor2k](https://github.com/naoto0804/cross-domain-detection) and [PeopleArt](https://github.com/BathVisArtData/PeopleArt) datasets **


### Installation
1. Clone the repository
  ```Shell
  git clone git@github.com:nicaogr/Mi_max.git
  ```

2. You need to install all the required python library such as tensorflow, cython, opencv-python, numpy, cython_bbox and easydict. We advice you to create a [conda environnement](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or equivalent.

You can use the requirements.txt file and then install [cython_bbox](https://pypi.org/project/cython-bbox/) :
  ```Shell
  pip install -r requirements.txt
  pip install cython_bbox
  ```
If you don't have the admin right try :
  ```Shell
  pip install --user -r requirements.txt
  pip install --user cython_bbox
  ```
In the requirements file, we install the GPU version of Tensorflow.
  
3. Update your -arch in setup script to match your GPU for the faster RCNN code. This code is a modify version of the [Xinlei Chen](https://github.com/endernewton) [implementation](https://github.com/endernewton/tf-faster-rcnn).
  ```Shell
  cd tf_faster_rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.

4. Build the Cython modules
If it is not already the case move to the lib folder.
  ```Shell
  cd tf_faster_rcnn/lib 
  make clean
  make
  cd ../../
  ```

This code have been tested on Ubuntu 18.04 and 16.04 with python 3.6 and Tensorflow 1.8 .

### Download the pre-trained models
You have to download the pre-trained models through the link below and saved them to the model folder after decompressed it :
  - Google drive [here](https://drive.google.com/open?id=0B1_fAEgxdnvJSmF3YUlZcHFqWTQ).
  
Here the matching between the network and training set and the ckpt weights file name, the last one is the one used by default in our code and our research paper : 
  | -------------------------- | -------------------------- |
  |   vgg16_VOC07  |  vgg16_faster_rcnn_iter_70000.ckpt  | 
  |   vgg16_VOC12  |  vgg16_faster_rcnn_iter_110000.ckpt  | 
  |   vgg16_COCO  |  vgg16_faster_rcnn_iter_1190000.ckpt  | 
  |   res101_VOC07  | res101_faster_rcnn_iter_70000.ckpt  | 
  |   res101_VOC12  |  res101_faster_rcnn_iter_110000.ckpt  | 
  |   res101_COCO  |  res101_faster_rcnn_iter_1190000.ckpt  | 
  |   res152_COCO  |  res152_faster_rcnn_iter_1190000.ckpt | 

### You may need to download the images datasets : (even if the 

You have to unzip the dataset in the data folder.

 - [IconArt](https://wsoda.telecom-paristech.fr/downloads/dataset/IconArt_v1.zip)
 - [PeopleArt](https://codeload.github.com/BathVisArtData/PeopleArt/zip/master) with the [image level labels](https://wsoda.telecom-paristech.fr/downloads/dataset/PeopleArt.csv)
 - [Watercolor2k](http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/watercolor.zip)  with the [image level labels](https://wsoda.telecom-paristech.fr/downloads/dataset/watercolor.csv)

### Description of the pipeline :

We will first download the required dataset, compute the features and boxes from a Faster RCNN (recorded in a tfrecords file) and then train a MI_max model before runnning the detection evaluation. The precomputation may require several dozen of GB.

You can add your own dataset, it only required to be in the PASCAL VOC format.

### To run the experiment from our research paper :
For the MI_max model :
  ```Shell
  python Run_MImax.py --dataset IconArt_v1 --with_score
  ```

For the MI_max-C model (the dataset is split in 2 and several different values for C are used) : 
  ```Shell
  python Run_MImax.py --dataset IconArt_v1 --with_score --CV_Mode CV --C_Searching
  ```

In the Run_MImax.py , you can find other parameters of the model that you can change such as the learning rate (LR), the regularization term (C), the loss ('' or 'hinge'), the number of restarts etc.

### Remark

The training part of the MI_max is trained in around 6 minutes on a good consumer GPU but we are sure that it is possible to speed it up.

Don't hesitate to contact us, if you have any question or remarks about our code and our work.

### Citation
Please consider citing:

    @article{Gonthier18,
         author       = "Gonthier, N. and Gousseau, Y. and Ladjal, S. and Bonfait, O.",
         title        = "Weakly Supervised Object Detection in Artworks",
         booktitle    = "Computer Vision -- ECCV 2018 Workshops",
         year         = "2018",
         publisher    = "Springer International Publishing",
         pages        = "692--709"
    }
