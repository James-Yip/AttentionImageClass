import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import torch
from math import *

# hyperparameters
alpha   = 0.5
beta    = 0.1
lambda1 = 0.01
lambda2 = 0.1
gama    = 0.1


def getAnchorPoints(num_points):
    radius = 0.5 * sqrt(2)
    # difference between two anchor points
    diff = 2 * pi / num_points
    cx = [radius * cos(i * diff) for i in range(0, num_points)]
    cy = [radius * sin(i * diff) for i in range(0, num_points)]

    return tensor(cx).view(num_points, -1), tensor(cy).view(num_points, -1)


'''
Loss Function for AttentionImageClass
=======================================================================================
@Args:
    input  : score vectors         (torch.cuda.FloatTensor[batch_size, num_categories])
    target : target                (torch.cuda.FloatTensor[batch_size, num_categories])
    M      : transformation matrix (torch.FloatTensor[num_iterations, batch_size, 2, 3])

@Returns:
    total_loss
=======================================================================================
'''
def loss_function(input, target, M, add_constraint=False):
    '''
    [variable] 'pp'       : predicted probability vector    
    [variable] 'gtp'      : ground-truth probability vector
    [variable] 'loss_cls' : loss for classification
    [variable] 'loss_loc' : loss for localizatoin
    '''
    
    # extra arguments from theta(that is transformation matrix)
    # =========================================================
    sx = M[1:, :, 0, 0]
    sy = M[1:, :, 1, 1]
    tx = M[1:, :, 0, 2]
    ty = M[1:, :, 1, 2]
     
    # anchor point
    # ============
    cx = tensor([0., 0.5,  0.5, -0.5, -0.5]).view(5, -1)
    cy = tensor([0., 0.5, -0.5,  0.5, -0.5]).view(5, -1)
    #cx, cy = getAnchorPoints(M.size(0) - 2)

    # calculate the predicted & ground-truth iprobability vector
    # ==========================================================
    pp  = F.softmax(input, dim=1)
    gtp = target.div(target.norm(p=1, dim=1).view(input.size()[0], -1))
    
    # calculate loss for classification
    # =================================
    loss_cls = F.mse_loss(pp, gtp, size_average=False)
    
    if not add_constraint:
        return loss_cls

    # calculate loss for localization
    # ===============================
    # anchor constraint
    loss_A = torch.sum(0.5 * ((tx - cx)**2 + (ty - cy)**2))
    
    # scale constraint
    loss_sx = torch.sum(torch.max(abs(sx) - alpha, tensor(0.)) ** 2)
    loss_sy = torch.sum(torch.max(abs(sy) - alpha, tensor(0.)) ** 2)
    loss_S  = loss_sx + loss_sy 

    # positive constraint
    loss_P = torch.sum(torch.max(beta - sx, tensor(0.)) + torch.max(beta - sy, tensor(0.)))

    loss_loc = (loss_S + lambda1 * loss_A + lambda2 * loss_P).cuda()
    
    # calculate total loss
    # ====================
    total_loss = loss_cls + gama * loss_loc
    
    print("M ", M)
    #print('sx ', sx)
    #print('sy ', sy)
    #print('tx ', tx)
    #print('ty ', ty)

    #print('cx ', cx)
    #print('cy ', cy)
    
    print('loss_A ', loss_A)
    #print('loss_S ', loss_S)
    #print('loss_P ', loss_P)
    print('loss_loc ', loss_loc)
    print("loss_cls ", loss_cls)
    print('total_loss ', total_loss)
    
    return total_loss

