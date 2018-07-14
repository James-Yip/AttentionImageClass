import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import visdom
import torch.nn.functional as F
import PIL.Image as Image


import argparse
import os

from models.RMA_module_with_priori import RMA_module
from models.loss_with_priori import loss_function
from utils import get_target_transform as target_trans
from utils import id2label


# GPU setting
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")


# ==================================================================
# Constants
# ==================================================================
EPOCH         = 45            # number of times for each run-through
BATCH_SIZE    = 8             # number of images for each epoch
N             = 512           # size of input images (512 or 640)
TOPK          = 3             # top k highest-ranked labels  
GPU_IN_USE    = torch.cuda.is_available()  # whether using GPU
PATH_MODEL_PARAMS  = './params/params_with_priori.pkl'


# ==================================================================
# Parser Initialization
# ==================================================================
parser = argparse.ArgumentParser(description='Pytorch Implementation of ICCV2017_AttentionImageClass')
parser.add_argument('--testBatchSize',   default=BATCH_SIZE,        type=int,   help='testing batch size')
parser.add_argument('--pathModelParams', default=PATH_MODEL_PARAMS, type=str,   help='path of model parameters')
parser.add_argument('--loadModel',       default=True,              type=bool,  help='load model parameters')
args = parser.parse_args()


# ==================================================================
# Transforms for the Input Images
# ==================================================================
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transforms = transforms.Compose([
             transforms.Resize((N, N)), 
             transforms.ToTensor(),
             normalize
             ]) 
        

class RMA_model(object):
    # 构造函数里加载模型，比如 tensorflow 的 graph, sess 等
    def __init__(self):
        # prepare model
        print('\n***** Prepare Model *****')
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.extract_features = vgg16.features
        self.RMA = RMA_module(lstm_input_size=14, lstm_hidden_size=4096, zk_size=4096)
        if args.loadModel:
            self.RMA.load_state_dict(torch.load(args.pathModelParams))
        if GPU_IN_USE:
            print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
            print('cuda: move all model parameters and buffers to the GPU')
            self.extract_features.cuda()
            self.RMA.cuda()
            cudnn.benchmark = True
        print('Model Preparation : Finished')

    # Test
    def evaluate(self, data):
        print('evaluate:')
        self.RMA.eval()        # set the module in evaluation mode
        print('before transforms')
        data = transforms(data).unsqueeze(0)
        if GPU_IN_USE:
            data = data.cuda()  # set up GPU Tensor
        
        print('before extracting features')
        f_I = self.extract_features(data)        
        output, _ = self.RMA(f_I)
        print('after RMA')    
        prediction  = torch.topk(F.softmax(output, dim=1), 10, dim=1) 
        filter      = prediction[0].eq(0.1) + prediction[0].gt(0.1)
        category_id = torch.mul(prediction[1]+1, filter.type(torch.cuda.LongTensor))
        print(prediction[0])
        #print(category_id)
        return id2label(category_id)[0].tolist()


    # 需要对外提供一个 API，可以直接拿到你们的结果
    def image_recognition(self, image_path):
        # 业务逻辑
        print('image path: ', image_path)
        image = Image.open(image_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        with torch.no_grad():
            label = self.evaluate(image)
        print(label)
        return dict(
            data = label
        )
    
    def __del__(self):
        print("delete!")

# 生成模型实例
# 这里生成模型实例供 server 导入并调用
print("生成 RMA Model 实例.................")
RMA_model_instance = RMA_model()
print("RMA Model 实例生成完成...............")
