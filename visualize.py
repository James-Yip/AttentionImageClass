import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import visdom
import torch.nn.functional as F
from CocoDetection import CocoDetection
from visualizeImg import *
#from PIL import Image
#import torch

import argparse
import os

from models.RMA_module_with_priori import RMA_module
from models.loss_with_priori import loss_function
from utils import get_target_transform as target_trans


# GPU setting
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")


# ==================================================================
# Constants
# ==================================================================
EPOCH         = 45            # number of times for each run-through
BATCH_SIZE    = 16            # number of images for each epoch
N             = 512           # size of input images (512 or 640)
TOPK          = 3             # top k highest-ranked labels  
GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
DIR_TEST_IMAGES    = '../dataset/val2017/'
PATH_TEST_ANNFILE  = '../dataset/annotations/instances_val2017.json'
PATH_MODEL_PARAMS  = './params/params_with_priori.pkl'
NUM_CATEGORIES     = 80
OUTPUT_INTERVAL    = 100


# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation of ICCV2017_AttentionImageClass')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--pathModelParams', default=PATH_MODEL_PARAMS, type=str,   help='path of model parameters')
parser.add_argument('--loadModel',       default=True,              type=bool,  help='load model parameters')
args = parser.parse_args()


# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')

# transforms of test dataset
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
test_transforms = transforms.Compose([
                    transforms.Resize((N, N)), 
                    transforms.ToTensor(),
                  ]) 

test_dataset  = CocoDetection(root=DIR_TEST_IMAGES,  annFile=PATH_TEST_ANNFILE,
                              transform=test_transforms,  target_transform=target_trans)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=args.testBatchSize,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')

# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

vgg16 = torchvision.models.vgg16(pretrained=True)
extract_features = vgg16.features
RMA = RMA_module(lstm_input_size=14, lstm_hidden_size=4096, zk_size=4096)
if args.loadModel:
    RMA.load_state_dict(torch.load(args.pathModelParams))

if GPU_IN_USE:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    print('cuda: move all model parameters and buffers to the GPU')
    extract_features.cuda()
    RMA.cuda()
    cudnn.benchmark = True

print('Model Preparation : Finished')


# Test
# ================================================================================
# data:        [torch.cuda.FloatTensor of size [batch_size, 3, N, N] N=512/640]
# target:      [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# output:      [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# prediction: [
#              [torch.cuda.FloatTensor of size [batch_size, TOPK] (TOPK)],
#              [torch.cuda.LongTensor  of size [batch_size, TOPK] (index of TOPK)]
#             ]
# ================================================================================
def test():
    print('test:')
    #RMA.eval()        # set the module in evaluation mode

    sum_prediction_label         = torch.zeros(1, 80) + 1e-6
    sum_correct_prediction_label = torch.zeros(1, 80)
    sum_ground_truth_label       = torch.zeros(1, 80)

    for batch_num, (data, target, original_imgs) in enumerate(test_loader):
        if target.sum() == 0:
            continue
        target        = target.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        data          = data.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        original_imgs = original_imgs.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        
        #print('original_imgs ', original_imgs.size())

        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda()  # set up GPU Tensor

        f_I = extract_features(data)        
        output, M, scores = RMA(f_I, return_whole_scores=True)
        
        #total_thetas.append(M)
        #total_scores.append(scores)
        
        #visualize_attentional_regions(original_imgs, M[1:, :, :, :], scores)
        visualize_attentional_regions(original_imgs, M, scores)
        
        prediction    = torch.topk(F.softmax(output, dim=1), 10, dim=1) 
        filter        = prediction[0].eq(0.1) + prediction[0].gt(0.1)
        prediction_index         = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
        extend_eye_mat           = torch.cat((torch.zeros(1, 80), torch.eye(80)), 0)
        prediction_label         = extend_eye_mat[prediction_index.view(-1)].view(-1, 10, 80).sum(dim=1)
        correct_prediction_label = (target.cpu().byte() & prediction_label.byte()).type(torch.FloatTensor)
        
        #count the sum of label vector
        sum_prediction_label         += prediction_label.sum(dim=0)
        sum_correct_prediction_label += correct_prediction_label.sum(dim=0)
        sum_ground_truth_label       += target.cpu().sum(dim=0)
        
        #for i in range(0, target.size(0)):
        #    print('-----------------')
        #    print('ground-truth: ', target[i].nonzero().view(-1))
        #    print('prediction:   ', prediction_index[i] - 1)
        #    print('-----------------')
        
        if batch_num % OUTPUT_INTERVAL == 0:
            print(batch_num)
            #print('loss %.3f (batch %d)' % (test_loss / (batch_num+1), batch_num+1))
    
    #evaluation metrics
    o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = torch.div(2 * o_p * o_r, o_p + o_r)
    c_p = (torch.div(sum_correct_prediction_label, sum_prediction_label)).sum() / NUM_CATEGORIES
    c_r = (torch.div(sum_correct_prediction_label, sum_ground_truth_label)).sum() / NUM_CATEGORIES
    cf1 = torch.div(2 * c_p * c_r, c_p + c_r)
   
    print('-------------------------------------------------------------')
    print('|    CP   |    CR   |    CF1  |    OP   |    OR   |    OF1  |')
    print('-------------------------------------------------------------')
    print('|  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |' % (c_p, c_r, cf1, o_p, o_r, of1))
    print('-------------------------------------------------------------')
    


# ==================================================================
# Save Parameters of Test
# ==================================================================
#def save():
#    torch.save(RMA.state_dict(), args.pathModelParams)
#    print('Checkpoint saved to {}'.format(args.pathModelParams))


# ==================================================================
# Main
# ==================================================================
with torch.no_grad():
    test()

