import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import visdom
import torch.nn.functional as F

import argparse
import os

from models.RMA_module_with_priori import RMA_module
from models.loss_with_priori import loss_function
from utils import get_target_transform as target_trans
from utils import*

# data visualization
vis = visdom.Visdom(env='priori_tencrop')
# GPU setting
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

# ==================================================================
# Constants
# ==================================================================
EPOCH         = 45            # number of times for each run-through
BATCH_SIZE    = 16            # number of images for each epoch
LEARNING_RATE = 1e-5          # default learning rate 
WEIGHT_DECAY  = 0             # default weight decay
N             = 576           # size of input images (512 or 640)
MOMENTUM      = (0.9, 0.999)  # momentum in Adam optimization
TOPK          = 3             # top k highest-ranked labels  
GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
DIR_TRAIN_IMAGES   = '../dataset/train2017/'
DIR_TEST_IMAGES    = '../dataset/val2017/'
PATH_TRAIN_ANNFILE = '../dataset/annotations/instances_train2017.json'
PATH_TEST_ANNFILE  = '../dataset/annotations/instances_val2017.json'
PATH_MODEL_PARAMS  = './params/params_with_priori.pkl'
NUM_CATEGORIES     = 80
LOSS_OUTPUT_INTERVAL = 100
CROPSIZE_512 = 16

# ==================================================================
# Global Variables
# ==================================================================
# one iteration means one mini-batch finishs a forward-backward process
current_training_iteration = torch.tensor([1])
current_test_iteration     = torch.tensor([1])
loss_graph_window          = 'loss graph'
test_f1_graph_window       = 'test OF1 and CF1 graph'
evaluation_window          = 'six evaluation metrics'
#category_id_window         = 'category ids of prediction and ground-truth'
of1 = 0.
cf1 = 0.

# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation of ICCV2017_AttentionImageClass')
parser.add_argument('--lr',              default=LEARNING_RATE,     type=float, help='learning rate')
parser.add_argument('--epoch',           default=EPOCH,             type=int,   help='number of epochs')
parser.add_argument('--trainBatchSize',  default=BATCH_SIZE,        type=int,   help='training batch size')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--weightDecay',     default=WEIGHT_DECAY,      type=float, help='weight decay')
parser.add_argument('--pathModelParams', default=PATH_MODEL_PARAMS, type=str,   help='path of model parameters')
parser.add_argument('--saveModel',       default=True,              type=bool,  help='save model parameters')
parser.add_argument('--loadModel',       default=False,             type=bool,  help='load model parameters')
args = parser.parse_args()


# ==================================================================
# Prepare Dataset(training & test)
# ==================================================================
print('***** Prepare Data ******')

# transforms of training dataset 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                     transforms.Resize((N, N)),
                     transforms.ToTensor(),
                     normalize
                  ])

# transforms of test dataset
test_transforms = transforms.Compose([
                    transforms.Resize((N, N)), 
                    transforms.ToTensor(),
                    normalize
                  ]) 

train_dataset = torchvision.datasets.CocoDetection(root=DIR_TRAIN_IMAGES, annFile=PATH_TRAIN_ANNFILE, 
                                                   transform=train_transforms, target_transform=target_trans)
test_dataset  = torchvision.datasets.CocoDetection(root=DIR_TEST_IMAGES,  annFile=PATH_TEST_ANNFILE,
                                                   transform=test_transforms,  target_transform=target_trans)
train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.trainBatchSize, shuffle=True,  num_workers=2)
test_loader   = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=args.testBatchSize,  shuffle=False, num_workers=2)
print('Data Preparation : Finished')


# ==================================================================
# Prepare Model
# ==================================================================
print('\n***** Prepare Model *****')

vgg16 = torchvision.models.vgg16(pretrained=True)

for param in vgg16.features.parameters():
    param.requires_grad=False

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

# Adam optimization
optimizer = optim.Adam(RMA.parameters(), lr=args.lr, weight_decay=args.weightDecay, betas=MOMENTUM)  
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)  # lr decay
print('Model Preparation : Finished')


# Train
# ================================================================================
# data:        [torch.cuda.FloatTensor of size [batch_size, 3, N, N] N=512/640]
# target:      [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# output:      [torch.cuda.FloatTensor of size [batch_size, num_categories]]
# prediction: [
#              [torch.cuda.FloatTensor of size [batch_size, TOPK] (TOPK)],
#              [torch.cuda.LongTensor  of size [batch_size, TOPK] (index of TOPK)]
#             ]
# ================================================================================
def train():
    print('train:')
    RMA.train()     # set the module in training  mode
    train_loss = 0. # sum of train loss up to current batch

    global current_training_iteration
    
    sum_prediction_label         = torch.zeros(1, 80) + 1e-6
    sum_correct_prediction_label = torch.zeros(1, 80)
    sum_ground_truth_label       = torch.zeros(1, 80)
    
    for batch_num, (data, target) in enumerate(train_loader):
        if target.sum() == 0:
            continue
        target = target.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        data   = data.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        
        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda() 

        # -----forward-----
        optimizer.zero_grad()
        f_I = extract_features(data)        
        output, M = RMA(f_I)
        # ---end forward---
        
        # ---calculate loss and backward---
        loss = loss_function(output, target, M, add_constraint=True)
        loss.backward()
        optimizer.step()
        # ----------end backward-----------
        
        train_loss   += loss
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
        #    print('prediction:   ', prediction[1][i])
        #    print('-----------------')
        
        if batch_num % LOSS_OUTPUT_INTERVAL == 0:
            # visualization: draw the train loss graph 
            vis.line(
                X=current_training_iteration, 
                Y=torch.tensor([train_loss.data]) / (batch_num+1), 
                win=loss_graph_window,
                name='train loss',
                update=None if current_training_iteration == 1 else 'append',
                opts=dict(xlabel='iteration', ylabel='loss', showlegend=True)
            )
            print('loss %.3f (batch %d)' % (train_loss/(batch_num+1), batch_num+1))
            current_training_iteration += LOSS_OUTPUT_INTERVAL

    # evaluation metrics
    o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = torch.div(2 * o_p * o_r, o_p + o_r)
    c_p = (torch.div(sum_correct_prediction_label, sum_prediction_label)).sum() / NUM_CATEGORIES
    c_r = (torch.div(sum_correct_prediction_label, sum_ground_truth_label)).sum() / NUM_CATEGORIES
    cf1 = torch.div(2 * c_p * c_r, c_p + c_r)
   
    return c_p, c_r, cf1, o_p, o_r, of1


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
    RMA.eval()        # set the module in evaluation mode
    test_loss    = 0. # sum of train loss up to current batch

    global current_test_iteration
    
    sum_prediction_label         = torch.zeros(1, 80) + 1e-6
    sum_correct_prediction_label = torch.zeros(1, 80)
    sum_ground_truth_label       = torch.zeros(1, 80)

    for batch_num, (data, target) in enumerate(test_loader):
        if target.sum() == 0:
            continue
        target = target.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))
        data   = data.index_select(0, torch.nonzero(target.sum(dim=1)).view(-1))

        if GPU_IN_USE:
            data, target = data.cuda(), target.cuda()  # set up GPU Tensor

        f_I = extract_features(data)   
        # ten-crop
        # f_I: batchsize*channel*inputSize*inputSize
        # tencrop_results: 10*batchsize*channel*cropSize*cropSize
        tencrop_results = tencrop(f_I, f_I.size(0), f_I.size(1), f_I.size(2), CROPSIZE_512)
        RMA_outputs = torch.zeros(target.size())
        RMA_losses = 0
        tencrop_results = tencrop_results.cuda()
        RMA_outputs = RMA_outputs.cuda()

        for i in range(10):
            crop_RMA_output, crop_RMA_M = RMA(tencrop_results[i])
            RMA_outputs += crop_RMA_output
            RMA_losses  += loss_function(crop_RMA_output, target, crop_RMA_M, add_constraint=True)

        output = RMA_outputs * 0.1
        loss = RMA_losses * 0.1     

        # output, M = RMA(f_I)
        # loss = loss_function(output, target, M, add_constraint=True)
        
        test_loss    += loss
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
        # 

        if batch_num % LOSS_OUTPUT_INTERVAL == 0:
            # visualization: draw the test loss graph
            vis.line(
                X=current_test_iteration, 
                Y=torch.tensor([test_loss.data]) / (batch_num+1), 
                win=loss_graph_window,
                name='test loss',
                update=None if current_test_iteration == 1 else 'append',
                # update='insert' if current_test_iteration == 1 else 'append',
                opts=dict(showlegend=True),
            )
            print('loss %.3f (batch %d)' % (test_loss / (batch_num+1), batch_num+1))
            current_test_iteration += LOSS_OUTPUT_INTERVAL

    # evaluation metrics
    o_p = torch.div(sum_correct_prediction_label.sum(), sum_prediction_label.sum())
    o_r = torch.div(sum_correct_prediction_label.sum(), sum_ground_truth_label.sum())
    of1 = torch.div(2 * o_p * o_r, o_p + o_r)
    c_p = (torch.div(sum_correct_prediction_label, sum_prediction_label)).sum() / NUM_CATEGORIES
    c_r = (torch.div(sum_correct_prediction_label, sum_ground_truth_label)).sum() / NUM_CATEGORIES
    cf1 = torch.div(2 * c_p * c_r, c_p + c_r)
   
    return c_p, c_r, cf1, o_p, o_r, of1


# ==================================================================
# Save Model
# ==================================================================
def save():
    torch.save(RMA.state_dict(), args.pathModelParams)
    print('Checkpoint saved to {}'.format(args.pathModelParams))


# ==================================================================
# Main Loop
# ==================================================================
for current_epoch in range(1, args.epoch + 1):
    print('\n===> epoch: %d/%d' % (current_epoch, args.epoch))
    # train_cp, train_cr, train_cf1, train_op, train_or, train_of1 = train()
    with torch.no_grad():
        test_cp, test_cr, test_cf1, test_op, test_or, test_of1 = test()
    
    evaluation_metrics = '''
<pre>
===> epoch: %d/%d<br/>
-------------------------------------------------------------
|    CP   |    CR   |   CF1   |    OP   |    OR   |   OF1   |
-------------------------------------------------------------
|  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |  %.3f  |
-------------------------------------------------------------
</pre>
''' % (current_epoch, args.epoch, test_cp, test_cr, test_cf1, test_op, test_or, test_of1)
    
    # visualization
    vis.line(
        X=torch.tensor([current_epoch]), 
        Y=torch.tensor([test_cf1]),
        name='test_CF1',
        win=test_f1_graph_window,
        update=None if current_epoch == 1 else 'append', 
        opts=dict(xlabel='epoch', ylabel='F1', showlegend=True, title='Evaluation of Test (CF1 / OF1)')
    )
    vis.line(
        X=torch.tensor([current_epoch]), 
        Y=torch.tensor([test_of1]),
        name='test_OF1',
        win=test_f1_graph_window,
        update='insert' if current_epoch == 1 else 'append', 
        opts=dict(showlegend=True)
    )
    vis.text(
        evaluation_metrics,
        win=evaluation_window,
        append=False if current_epoch == 1 else True
    )
    
    if test_of1 > of1 and test_cf1 > cf1:
        if args.saveModel:
            save()
        of1 = test_of1
        cf1 = test_cf1

    if current_epoch == args.epoch:
        print('===> BEST PERFORMANCE (OF1/CF1): %.3f / %.3f' % (of1, cf1))

