import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor
import torch

COCO_CATEGORIES = 80

'''
Recurrent Memorized-Attention Module
==============================================================================================
@Parameters:
    lstm_input_size  : number of expected features in the input x of LSTM
    lstm_hidden_size : number of features in the hidden state of LSTM
    zk_size          : size of z_k (about z_k, see 'Update rule of M' in the paper)
    num_itreations   : number of iterations in RMA module (default: 5)
    num_classes      : number of classes/categories (default: 80, using COCO dataset)
    use_gpu          : whether using gpu (default: True)
@Input:
    f_I : feature map (torch.cuda.FloatTensor[batch_size, num_channels, height, width])
@Output:
    fused_scores : final fused score vectors (torch.cuda.FloatTensor[batch_size, num_classes])
    M            : transformation matrices in ST
==============================================================================================
'''
class RMA_module(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, zk_size,
                 num_iterations=5, num_classes=COCO_CATEGORIES,
                 use_gpu=True):
        
        super(RMA_module, self).__init__()

        self.K           = num_iterations 
        self.C           = num_classes
        self.use_gpu     = use_gpu
        self.input_size  = lstm_input_size
        self.hidden_size = lstm_hidden_size

        self.cx = tensor([0.5,  0.5, -0.5, -0.5]).view(4, -1)
        self.cy = tensor([0.5, -0.5,  0.5, -0.5]).view(4, -1)
        
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.fc      = nn.Linear(lstm_input_size * lstm_input_size / 4 * 512, 4096)
        self.lstm = nn.LSTMCell(4096, lstm_hidden_size)
        
        self.get_zk = nn.Sequential(
            # channels of output feature map in vgg16 = 512
            nn.Linear(lstm_hidden_size, zk_size),     
            nn.ReLU(inplace=True)
        )
        self.get_score = nn.Linear(zk_size, num_classes)
        self.update_m  = nn.Linear(zk_size, 6)
        self.update_m.weight.data = torch.zeros(6, zk_size)
        self.update_m.bias.data   = tensor([1., 0., 0., 0., 1., 0.])

    # ST: spatial transformer network forward function
    # ================================================
    def ST(self, x, theta, k):
        # determine the output size of STN
        num_channels = x.size()[1]
        batch_size   = x.size()[0]
        output_size =  torch.Size((batch_size, num_channels, self.input_size, self.input_size))
        
        if k > 1:
            theta[:, 0, 2] = theta[:, 0, 2] + self.cx[k-2]
            theta[:, 1, 2] = theta[:, 1, 2] + self.cy[k-2]

        grid = F.affine_grid(theta, output_size) 
        if self.use_gpu:
            grid = grid.cuda()
        # use bilinear interpolation(default) to sample the input pixels
        x = F.grid_sample(x, grid)
        return x, theta
    
    # init_hidden: initialize the (h0, c0) in LSTM
    # ============================================
    def init_hidden(self, N):
        if self.use_gpu:
            h0 = torch.zeros(N, self.hidden_size).cuda()
            c0 = torch.zeros(N, self.hidden_size).cuda()
        else:
            h0 = torch.zeros(N, self.hidden_size)
            c0 = torch.zeros(N, self.hidden_size)
        return (h0, c0)
    
    # RMA moudule forward function
    # ============================
    def forward(self, f_I, return_whole_scores=False):
        # initialization
        batch_size = f_I.size()[0]
        hidden = self.init_hidden(batch_size)
        if self.use_gpu:
            scores = torch.randn(self.K, batch_size, self.C).cuda()
        else:
            scores = torch.randn(self.K, batch_size, self.C)
        M      = torch.randn(self.K+1, batch_size, 2, 3) 
        M[0]   = tensor([[1., 0., 0.], [0., 1., 0.]])
        M_for_visual = torch.randn(self.K, batch_size, 2, 3)

        # for each iteration
        for k in range(0, self.K+1):
            # locate an attentional region
            f_k, M_k_for_visual = self.ST(f_I, M[k].clone(), k)
            
            # descend dimension for lower GPU memory requirement
            f_k = self.pooling(f_k)
            f_k = self.fc(f_k.view(batch_size, -1))

            # predict the scores regarding this region
            hidden = self.lstm(f_k, hidden)
            
            # get z_k for further caculating M and scores
            z_k = self.get_zk(hidden[0])

            if k != 0:
                # obtain the score vector of current iteration
                scores[k-1] = self.get_score(z_k)
                M_for_visual[k-1] = M_k_for_visual
                
            if k != self.K:
                # update transformation matrix for next iteration
                M[k+1] = self.update_m(z_k).view(batch_size, 2, 3)
                M[k+1, :, 0, 1] = tensor(0.)
                M[k+1, :, 1, 0] = tensor(0.)
        
        # max pooling to obtain the final fused scores
        fused_scores = scores.max(0)
       
        #print('M for visual ', M_for_visual)

        if return_whole_scores:
            return fused_scores[0], M_for_visual, scores
        else:
            return fused_scores[0], M

