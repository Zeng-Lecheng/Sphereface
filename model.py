import torch
import torch.nn as nn
from torch import pi, cos
from torch.nn.parameter import Parameter


def phi(theta, m):
    k = theta // (pi / m)
    return (-1) ** k * cos(m * theta)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)    # weight init
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # ||x|| size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # ||ww|| size=Classnum

        cos_theta = x.mm(ww) # ww.x size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)   # ww.x / (||x|| ||w||)
        cos_theta = cos_theta.clamp(-1, 1)   # ?

        # cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = torch.acos(cos_theta)
        cos_m_theta = torch.cos(self.m * theta)
        k = torch.floor(self.m * theta / pi)
        phi_theta = (-1**k) * cos_m_theta # - 2*k

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        return cos_theta, phi_theta # size=(B,Classnum,2)


class Net(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.num_class = num_class

        self.conv1 = nn.Conv2d(3, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)

        self.relu1 = nn.PReLU(64)
        self.relu2 = nn.PReLU(128)
        self.relu3 = nn.PReLU(256)
        self.relu4 = nn.PReLU(512)

        self.fc1 = nn.Linear(100352, 512)
        self.fc2 = AngleLinear(512, self.num_class)

    def forward(self, x, get_feature=False):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        if get_feature:
            return x
        else:
            return self.fc2(x)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1) #size=(B,1)

        index = torch.zeros(cos_theta.shape, dtype=torch.bool).to(cos_theta.device) #size=(B,Classnum)
        index.scatter_(1, target.data, 1)

        self.lamb = max(self.LambdaMin, self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = torch.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss
