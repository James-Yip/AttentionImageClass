import torch
import numpy as np
from numpy import *
import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch.nn.functional as F

CLASS = np.array(['None', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])


def get_target_transform(target):
    labelsmap = {}
    target_transform = torch.zeros(80)
    labels = open('coco-label.txt', 'r')
    for line in labels:
        ids = line.split(',')
        labelsmap[int(ids[0])] = int(ids[1])
    for obj in target:
        if 'category_id' in obj:
            catId = obj['category_id']
            target_transform[labelsmap[catId] - 1] = 1
    #print(target[0]['image_id'])
    #print(target['image']['id']) 
    return target_transform

 # inputs: batchsize*channel*inputSize*inputSize(tensor)
 # batchSize = 8
 # channel   =
 # inputSize =
 # cropSize  =
 # return: 10*batchsize*channel*cropSize*cropSize(tensor)
def tencrop(inputs, batchSize, channel, inputSize, cropSize):
    crops = torch.zeros(10, batchSize, channel, cropSize, cropSize)
    crops = crops.numpy()
    edgestart = inputSize-cropSize;
    midstart = int(floor(inputSize/2)-floor(cropSize/2))
    midend = midstart+cropSize;

    crops[0] = inputs[:,:,0:cropSize,0:cropSize]
    crops[1] = inputs[:,:,0:cropSize,edgestart:inputSize]
    crops[2] = inputs[:,:,edgestart:inputSize,0:cropSize]
    crops[3] = inputs[:,:,edgestart:inputSize,edgestart:inputSize]
    crops[4] = inputs[:,:,midstart:midend,midstart:midend]
    crops[5] = crops[0][:,:,:,::-1]
    crops[6] = crops[1][:,:,:,::-1]
    crops[7] = crops[2][:,:,:,::-1]
    crops[8] = crops[3][:,:,:,::-1]
    crops[9] = crops[4][:,:,:,::-1]
    crops = torch.from_numpy(crops)
    return crops


# input: 'scores' tensor[K, batch_size, num_categories]
# return : 'confidence' tensor[K, batch_size]
#          'className'  numpy.ndarray[K, batch_size]
def getPredictedInfo(scores): 
    confidence, category_id = torch.max(F.softmax(scores, dim=2), dim=2)
    #print(category_id)
    className = CLASS[category_id+1]
    return confidence, className


def id2label(category_id):
    return [CLASS[i.index_select(0, i.nonzero().view(-1))] for i in category_id]

