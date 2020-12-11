#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# Edited by Matthew Seals
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect

from torchvision.ops import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse
from matplotlib import cm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import pandas as pd                   #A library for handling CSVs and creating pandas dataframe. A dataframe is a datastructure used for CSVs in pandas

import torch

CLASSES = ('__background__', 'building', 'person', 'vehicle', 'weapon')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_%d.pth', ),
    'res101': ('res101_faster_rcnn_iter_%d.pth', )
}
DATASETS = {
    'pascal_voc': ('voc_2007_trainval', ),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval', )
}

COLORS = [cm.tab10(i) for i in np.linspace(0., 1., 10)]


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    lst=[]                                         #INITIALISATION OF LIST, IT WILL BE USED TO STORE ENTITIES LIKE IMAGE_NAME, OBJECT_CLASS, ETC
    count_target=0                                 #A COUNTER USED FOR THE CLASSES THAT ARE NOT PRESENT IN A PARTICULAR IMAGE, SO, IF IT IS FOUR THAT MEANS THE IMAGE HAS NO TARGET OBJECTS
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    fname=im_file.split('/')[-1]                   #CONVERTED THE WHOLE IMAGE PATH TO IMAGE FILENAME THAT IS NEEDED TO BE SPECIFIED IN THE OUTPUT CSV GENERATED.

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    thresh = 0.8  # CONF_THRESH
    NMS_THRESH = 0.3

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    cntr = -1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            count_target+=1                                      #INCREMENTING THE COUNT SO THAT NO. OF CLASSES NOT PRESENT CAN BE CALCULATED, AS IN THIS PART THE LENGTH OF inds OF
            if count_target==4:                                  #PARTICULAR OBJECT IS CHEKED IF IT IS ZERO, THEN continue IS USED SO THAT THE FURTHER DETECTION PROCESS COULD NOT BE DONE.
            	writer(lst,count_target,fname)					 #fname IS PROVIDED AS A PARAMETER SEPARATELY TO THE writer FUNCTION FOR THOSE IMAGES WHICH DONT HAVE ANY TARGET OBJECT/CLASS, FOR IMAGES HAVING THE TARGET, A SEPARATE LIST 'a' IS CREATED BELOW.
            continue
        else:
            cntr += 1

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              fill=False,
                              edgecolor=COLORS[cntr % len(COLORS)],
                              linewidth=3.5))
            ax.text(
                bbox[0],
                bbox[1] - 2,
                '{:s} {:.3f}'.format(cls, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14,
                color='white')
            lst=[fname,cls,bbox[0],bbox[1],bbox[2],bbox[3],score]    #A LIST 'a' WHICH CONTAINS PARAMETERS TAKEN FROM THE DETECTED IMAGE. IT HAS PARAMETERS LIKE FILENAME, CLASS, BOUNDING BOX CO-ORD. IN THE FORM[x y w h] (x,y)->CO-ORD. OF TOP LEFT CORNER OF BBOX, w & h ARE THE WIDTH OF BBOX. WHETHER TARGET IS PRESENT OR NOT, IS NOT A PART OF THIS LIST AS ENTRIES OF THIS COLUMN ARE FILLED SEPARATELY INSIDE THE writer FN. ACCORDING TO THE CONDITION.
            writer(lst,count_target,fname)						   #writer FUNCTION IS CALLED AND LIST(a) IS PASSED AS A PARAMETER TO IT
        ax.set_title(
            'All detections with threshold >= {:.1f}'.format(thresh),
            fontsize=14)

        plt.axis('off')
        plt.tight_layout()
    plt.savefig(os.path.join('img_results', 'demo_' + image_name))
    if count_target==4:
    	print('No target objects present')
    print('Saved to `{}`'.format(
        os.path.join(os.getcwd(), 'img_results', 'demo_' + image_name)))

def writer(lst,count_target,fname):                                 #THIS FUNCTION IS USED TO POPULATE THE FILE 'output_info.csv'
	with open("output_info.csv", "a") as f:                       #A FILE 'output_info.csv' IS CREATED AND ENTRIES ARE FILLED USING THE LIST a,  CREATED EARLIER IN demo FUNCTION.
		if count_target==4:                                       #ONLY THE 'TARGET IS PRESENT OR NOT'(specified by target with boolean entries, in output_info.csv) AND FNAME IS FILLED FOR IMAGES WITH NO TARGET OBJECTS(which can be detected using the condition of count_target).
			f.write("0 %s\n" % fname)                             
		else:
			f.write("1 ")
			for ent in lst:
				f.write("%s " % ent)
				if (ent==lst[-1]):
					f.write("\n")
	f.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument(
        '--net',
        dest='demo_net',
        help='Network to use [vgg16 res101]',
        choices=NETS.keys(),
        default='res101')
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Trained dataset [pascal_voc pascal_voc_0712]',
        choices=DATASETS.keys(),
        default='pascal_voc')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    if os.path.isfile("output_info.csv"):
    	os.remove('output_info.csv')                                                           #FOR DELETING PREVIOUSLY CREATED FILE 
    col=['target','image_name','class','bbox_x','bbox_y','bbox_w','bbox_h','conf_score']   #HERE COLUMNS OF THE RESULTING CSV FILE ARE BEING ADDED, WHICH WILL BE CONTAINING THE INFO RELATED TO THE IMAGES ON WHICH PREDICTIONS ARE BEING DONE
    with open("output_info.csv", "a") as f:										           #THe CSV FILE 'output_info' WILL BE POULATED LATER ON, USING THE writer FUNCTION.
    	for ent in col:	
    		f.write("%s " % ent)
    	if ent==col[-1]:
    		f.write("\n")

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    saved_model = os.path.join(
        'output', demonet, DATASETS[dataset][0], 'default',
        NETS[demonet][0] % (110000 if dataset == 'pascal_voc' else 150000))

    if not os.path.isfile(saved_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    #net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])
    net.create_architecture(5, tag='default', anchor_scales=[8, 16, 32])                  #IN OUR CASE THERE ARE 5 CLASSES. 1 IS __background__ class. So, 4+1=5

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    im_names = [
        i for i in os.listdir('data/demo/')  # Pull in all jpgs
        if i.lower().endswith(".jpg")
    ]

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

    plt.show()
