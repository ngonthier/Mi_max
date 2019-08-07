# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


from ..datasets.pascal_voc import pascal_voc
from ..datasets.CrossMod_db import CrossMod_db
from ..datasets.WikiTenLabels_db import WikiTenLabels_db
from ..datasets.IconArt import IconArt_v1
#from ..datasets.coco import coco # Commented by Nicolas because API COCO Python need python27 : it need to be modified problem with _mask

def get_sets(data_path='/media/gonthier/HDD/data/'):
  __sets = {}
  """Get an imdb (image database) by name.
  @param : data_path : localisation of the dataset
  """
  __sets = {}
  # Set up voc_<year>_<split> 
  for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,devkit_path=os.path.join(data_path,'VOCdevkit'),test_ext=True))
    
  for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_{}_{}_diff'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

  for db in ['watercolor','comic','clipart']:
    for split in ['train', 'test']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: CrossMod_db(db,split,devkit_path=os.path.join(data_path,'cross-domain-detection','datasets'),test_ext=True))
 
  for db in ['PeopleArt']:
    for split in ['train', 'test','trainval','val']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: CrossMod_db(db,split,devkit_path=data_path,test_ext=True))
 
  for db in ['WikiTenLabels']:
    for split in ['test']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: WikiTenLabels_db(db,split,devkit_path=os.path.join(data_path,'Wikidata_Paintings'),test_ext=True))
        
  for db in ['IconArt_v1']:
    for split in ['test','train']:
        name = '{}_{}'.format(db,split)
        __sets[name] = (lambda split=split, db=db: IconArt_v1(db,split,devkit_path=os.path.join(data_path,'Wikidata_Paintings'),test_ext=True))
 
## Set up coco_2014_<split>
#for year in ['2014']:
  #for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    #name = 'coco_{}_{}'.format(year, split)
    #__sets[name] = (lambda split=split, year=year: coco(split, year))

## Set up coco_2015_<split>
#for year in ['2015']:
  #for split in ['test', 'test-dev']:
    #name = 'coco_{}_{}'.format(year, split)
    #__sets[name] = (lambda split=split, year=year: coco(split, year))

  return(__sets)

def get_imdb(name,data_path='/media/gonthier/HDD/data/',ext=None):
  """Get an imdb (image database) by name.
  @param : data_path : localisation of the dataset
  """
  if not(os.path.exists(data_path)):
    data_path = 'data/'
  __sets = get_sets(data_path=data_path)
  
  if name not in __sets or not(ext is None):
    if not(ext is None) and 'IconArt_v1' in name:
      for split in ['test']:
        name = '{}_{}'.format('IconArt_v1',split)
        __sets[name] = (lambda split=split, db='IconArt_v1': IconArt_v1('IconArt_v1',split,devkit_path=os.path.join(data_path,'Wikidata_Paintings'),test_ext=True,ext=ext))    
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  __sets = get_sets()
  return list(__sets.keys())
