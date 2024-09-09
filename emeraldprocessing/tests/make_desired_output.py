#!/usr/bin/env python
# coding: utf-8

import emeraldprocessing.pipeline
import os
import yaml
from emeraldprocessing.tests.pipeline import reorder_xyz

datdir = 'demo_data'
project_crs = 31984   # demo data is part of Ancieta
root_dir = emeraldprocessing.pipeline.__file__.split('emeraldprocessing')[0]
os.chdir(root_dir)

def get_items_from_processing_list(processing_list, pattern):
    item_list=[]
    for item in processing_list:
        #if pattern in list(item.keys())[0]:
        if pattern.__eq__(list(item.keys())[0].split('.')[-1]):
            item_list.append(item)
    if len(item_list)>0:
        return item_list
    else:
        raise Exception('no item with pattern ´{}´ found in processsing list'.format(pattern))

def get_desired_output_file_names(datdir, prefix):
    output_dir=os.path.join(datdir, 'output')
    output_xyz=os.path.join(output_dir, prefix+'.xyz')
    output_alc=os.path.join(output_dir, prefix+'.alc')
    output_gex=os.path.join(output_dir, prefix+'.gex')
    return output_xyz, output_alc, output_gex

def load():
    return emeraldprocessing.pipeline.ProcessingData(os.path.join(datdir,'Demo.xyz'),
                                                     os.path.join(datdir,'SkyTEM_LMZ_HMZ_STD_demo.alc'),
                                                     os.path.join(datdir,'Demo.gex'),
                                                     crs=project_crs)


with open(os.path.join(datdir, 'processing_flow.yml'), 'r') as file:
    processing_list = yaml.safe_load(file)

d = load()
d.process(processing_list)
reorder_xyz(d.xyz)
d.save(os.path.join(datdir, "output/full_pipeline.xyz"), 
       os.path.join(datdir, "output/full_pipeline.alc"), 
       os.path.join(datdir, "output/full_pipeline.gex"))


for item in processing_list:
    task=list(item.keys())[0].split('.')[-1]
    processing_sublist = get_items_from_processing_list(processing_list, task)
    print('task: {0} options: \n {1}'.format(task, processing_sublist))
    
    d = load()
    d.process(processing_sublist)
    desired_xyz, desired_alc, desired_gex = get_desired_output_file_names(datdir, task)
    reorder_xyz(d.xyz)
    d.save(desired_xyz, desired_alc, desired_gex)
