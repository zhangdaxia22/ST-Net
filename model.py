import torch
from torch import nn
import numpy as np

class trans(nn.Module):
    def __init__(self,dim):
        super(trans, self).__init__()
        self.dim = dim
        self.mha = nn.MultiheadAttention(dim,1)
        self.ln_1 = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        self.ln_2 = nn.LayerNorm(dim)
    def  forward(self,x):
        x = x+ self.ln_1( self.mha(x,x,x)[0])
        x = x+self.ln_2(self.linear(x))

        return x


class Position_Embedding(nn.Module):
    def __init__(self):
        super(Position_Embedding, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, h, w = x.size()
        h_avg = torch.mean(x, dim=1)
        w_avg = torch.mean(x, dim=2)

        hh = h_avg.repeat(1, h, 1).reshape(b, 1, h, w)
        ww = w_avg.repeat(1, 1, w).reshape(b, 1, x.size(1), x.size(2))
        h_w = torch.cat([hh, ww], dim=1)

        return self.sig(self.conv1(h_w))


class conv(nn.Module):
    def __init__(self):
        super(conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,2,(3,1),stride=1,padding=(1,0)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2,1,(1,1),stride=1),
            nn.BatchNorm2d(1)
        )
    def forward(self,x):
        return self.conv1(x)
class CT(nn.Module):
    def __init__(self,dim):
        super(CT, self).__init__()
        self.ti_trans = trans(32)
        self.sp_trans = trans(dim)
        self.conv = conv()
        self.pos = Position_Embedding()
        self.pool = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        self.ln = nn.LayerNorm(dim)
        self.lamda = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self,x):

        x = self.conv(x)
        x = torch.squeeze(x)
        x = x*self.pos(x).squeeze(1)


        sp_x = self.sp_trans(x)
        ti_x = x.transpose(1,2)
        ti_x = self.ti_trans(ti_x)
        ti_x = ti_x.transpose(1,2)

        # x = sp_x+ti_x
        x = sp_x+self.lamda*ti_x

        x = self.ln(x)


        x = self.pool(x)
        x = x.unsqueeze(1)
        return x
class modelD(nn.Module):
    def __init__(self,dim,classier):
        super(modelD, self).__init__()
        self.block1 = CT(dim)
        self.block2 = CT(dim//2)
        self.block3 = CT(dim//4)
        self.block4 = CT(dim//8)
        # self.block5 = CT(dim//16)
        # self.block6 = CT(dim//32)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(32,classier)  # 多受试者分类器
        # self.linear = nn.Sequential(   # 单受试者分类器
        #     nn.Linear(32, 32),
        #     nn.Tanh(),
        #     nn.BatchNorm1d(32),
        #     nn.Linear(32, classier),
        # )
        # self.softmax = nn.Softmax(dim=1)

    def  forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)

       # torch.Size([64, 1, 32, 4])

        x = self.pool(x.squeeze(1))
        x = x.view(x.size(0),-1)
        x = self.linear(x)

        return x

batch_size = 64  # 假设我们的批次大小为64
eeg_batch = torch.rand((batch_size,1, 32, 128))
modelct = modelD(128,2)
outct = modelct(eeg_batch)
print(outct.shape)
